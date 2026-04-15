import base64
import json
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

FITZ_IMPORT_ERROR = None
try:
    import pymupdf as fitz  # Preferred modern import for PyMuPDF
except Exception:
    try:
        import fitz  # Backward compatible import
    except Exception as exc:
        fitz = None
        FITZ_IMPORT_ERROR = str(exc)

BASE_DIR = Path(__file__).resolve().parent
DOTENV_PATH = BASE_DIR / ".env"
DOTENV_FALLBACK_PATH = BASE_DIR / ".env.txt"

load_dotenv(dotenv_path=DOTENV_PATH, override=False)
if not os.getenv("GROQ_API_KEY", "").strip() and DOTENV_FALLBACK_PATH.exists():
    # Windows sometimes creates .env.txt by mistake from Notepad.
    load_dotenv(dotenv_path=DOTENV_FALLBACK_PATH, override=False)

st.set_page_config(page_title="Invoice Data Extractor", layout="wide")


def get_client():
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None
    return Groq(api_key=api_key)


def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")


def parse_json_response(response_text):
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    candidate = cleaned
    # Fallback if model returns text before/after JSON block
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]

    # Attempt 1: parse directly
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Attempt 2: repair common LLM JSON issues
    repaired = candidate
    repaired = repaired.replace("\t", " ")
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)  # trailing commas
    repaired = re.sub(r'([\{,]\s*)([A-Za-z_][A-Za-z0-9_\- ]*)(\s*:)', r'\1"\2"\3', repaired)
    repaired = re.sub(r":\s*'([^']*)'", r': "\1"', repaired)
    repaired = re.sub(r'"\s*:\s*"([^"]*)\n([^"]*)"', lambda m: f'" : "{m.group(1)} {m.group(2)}"', repaired)

    try:
        return json.loads(repaired)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned invalid JSON after repair attempts: {exc}") from exc


def repair_json_with_model(client, raw_text):
    prompt = f"""
Convert the following content into valid JSON.
Rules:
- Return ONLY valid JSON.
- Preserve keys and values as much as possible.
- Do not add markdown or explanations.

Content:
{raw_text}
"""
    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0,
    )
    fixed_text = completion.choices[0].message.content
    return parse_json_response(fixed_text)


def extract_data_from_images(client, image_bytes_list):
    prompt = """
Extract complete data from the provided invoice/receipt/document images.
Return ONLY valid JSON with this schema:
{
  "document_type": "",
  "vendor_name": "",
  "invoice_number": "",
  "invoice_date": "",
  "due_date": "",
  "currency": "",
  "subtotal": "",
  "tax": "",
  "discount": "",
  "shipping": "",
  "total_amount": "",
  "bill_to": "",
  "ship_to": "",
  "payment_terms": "",
  "po_number": "",
  "notes": "",
  "line_items": [
    {
      "item_id": "",
      "description": "",
      "quantity": "",
      "unit_price": "",
      "tax": "",
      "line_total": ""
    }
  ],
  "additional_fields": {}
}
Rules:
- Keep missing fields as empty strings.
- Do not use markdown code fences.
- Add any extra captured keys into additional_fields.
- CRITICAL: If item IDs/SKU/serial numbers are listed in one row (comma/newline separated),
  split them into separate line_items entries (one item_id per row).
- Do not return only summary like "33 items". Return each item_id separately.
- CRITICAL: For transporter invoices, item IDs may appear as dense comma-separated codes
  in the description block. Extract EVERY code exactly; do not skip repeated-looking IDs.
"""
    content_list = [{"type": "text", "text": prompt}]

    for img_bytes in image_bytes_list:
        content_list.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_bytes)}"},
            }
        )

    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": content_list}],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0,
    )
    response_text = completion.choices[0].message.content
    try:
        return parse_json_response(response_text)
    except ValueError:
        return repair_json_with_model(client, response_text)


def extract_item_ids_fallback(client, image_bytes_list, expected_count):
    if not client or expected_count <= 0:
        return []

    prompt = f"""
You are extracting transporter invoice item IDs only.
Return ONLY valid JSON with schema:
{{
  "item_ids": ["", ""]
}}
Rules:
- Extract item IDs/codes from the goods description block.
- Keep each code exactly as printed (letters/numbers/slash/hyphen only).
- Preserve source order.
- Return exactly {expected_count} IDs.
- Do not include quantity, HSN, invoice numbers, vehicle numbers, or GST values.
- No markdown code fences.
"""

    content_list = [{"type": "text", "text": prompt}]
    for img_bytes in image_bytes_list:
        content_list.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_bytes)}"},
            }
        )

    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": content_list}],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0,
    )
    response_text = completion.choices[0].message.content
    parsed = parse_json_response(response_text)
    raw_ids = parsed.get("item_ids", []) if isinstance(parsed, dict) else []
    if not isinstance(raw_ids, list):
        return []
    return clean_item_id_list(raw_ids)


def is_probable_item_id(candidate):
    if not candidate:
        return False
    if len(candidate) < 9:
        return False
    if not re.fullmatch(r"[A-Z0-9\-_\/]+", candidate):
        return False
    if not re.search(r"[A-Z]", candidate):
        return False
    if not re.search(r"\d", candidate):
        return False
    return True


def clean_item_id_list(values):
    cleaned = []
    seen = set()
    for value in values:
        candidate = str(value).strip().upper()
        candidate = re.sub(r"[^A-Z0-9\-_/,]", "", candidate)
        # Some model responses merge IDs in one string.
        parts = [p for p in re.split(r"[\s,;|]+", candidate) if p]
        for part in parts:
            part = part.strip()
            if not is_probable_item_id(part):
                continue
            if part in seen:
                continue
            seen.add(part)
            cleaned.append(part)
    return cleaned


def extract_item_ids_from_any_text(extracted_json):
    text_chunks = []

    def _walk(value):
        if isinstance(value, dict):
            for nested in value.values():
                _walk(nested)
        elif isinstance(value, list):
            for nested in value:
                _walk(nested)
        elif value is not None:
            text_chunks.append(str(value))

    _walk(extracted_json)
    if not text_chunks:
        return []

    combined_text = " ".join(text_chunks).upper()
    combined_text = re.sub(r"[^A-Z0-9,\-_/.\s]", " ", combined_text)
    pattern_hits = re.findall(r"\b[A-Z][A-Z0-9\-_\/]{8,}\b", combined_text)
    return clean_item_id_list(pattern_hits)


def prepare_images_from_upload(uploaded_file):
    file_extension = uploaded_file.name.split(".")[-1].lower()
    image_bytes_list = []

    if file_extension == "pdf":
        if fitz is None:
            st.error(
                "PDF processing is unavailable because PyMuPDF failed to import. "
                f"Details: {FITZ_IMPORT_ERROR}"
            )
            st.stop()

        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        pages_to_process = min(len(pdf_document), 5)
        st.info(f"Processing {pages_to_process} page(s) from PDF.")
        cols = st.columns(min(pages_to_process, 3))

        for i in range(pages_to_process):
            page = pdf_document[i]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_bytes = pix.tobytes("jpeg")
            image_bytes_list.append(img_bytes)

            with cols[i % len(cols)]:
                st.image(img_bytes, caption=f"Page {i + 1}", use_container_width=True)
    else:
        img_bytes = uploaded_file.read()
        image_bytes_list.append(img_bytes)
        st.image(img_bytes, caption="Uploaded Image", use_container_width=True)

    return image_bytes_list


def extract_item_ids_from_text(text_value):
    if not text_value:
        return []

    cleaned_text = str(text_value).upper()
    cleaned_text = cleaned_text.replace("\n", " ").replace("\r", " ")
    cleaned_text = re.sub(r"[\(\)\[\]]", " ", cleaned_text)
    cleaned_text = re.sub(r"[^A-Z0-9,\-_/.\s]", " ", cleaned_text)

    # First pass: tokenize by common separators.
    tokens = re.split(r"[\s,;|]+", cleaned_text)
    item_ids = []

    for token in tokens:
        candidate = token.strip().replace(".", "")
        if not candidate:
            continue
        if len(candidate) < 4:
            continue
        if not re.fullmatch(r"[A-Z0-9\-_\/]+", candidate):
            continue
        item_ids.append(candidate)

    # Fallback pass: if tokenization found nothing, capture contiguous IDs.
    if not item_ids:
        pattern_hits = re.findall(r"\b[A-Z0-9][A-Z0-9\-_]{3,}\b", cleaned_text)
        item_ids.extend(pattern_hits)

    return item_ids


def extract_numeric_amount(raw_value):
    if raw_value is None:
        return None

    text = str(raw_value).replace(",", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def format_amount(amount):
    if amount is None:
        return ""
    return f"{amount:.2f}"


def get_expected_item_count(line_items):
    if not isinstance(line_items, list):
        return None

    candidates = []
    for item in line_items:
        if not isinstance(item, dict):
            continue
        qty = extract_numeric_amount(item.get("quantity"))
        if qty is None:
            continue
        if qty > 1 and float(qty).is_integer():
            candidates.append(int(qty))

    if not candidates:
        return None
    return max(candidates)


def build_rows_from_ids(item_ids, template_item):
    if not item_ids:
        return []

    description = str(template_item.get("description", "")).strip()
    unit_price = str(template_item.get("unit_price", "")).strip()
    tax = str(template_item.get("tax", "")).strip()

    distributed_tax = tax
    parsed_tax = extract_numeric_amount(tax)
    if parsed_tax is not None:
        distributed_tax = format_amount(parsed_tax / len(item_ids))

    rows = []
    for item_id in item_ids:
        rows.append(
            {
                "item_id": item_id,
                "description": description,
                "quantity": "1",
                "unit_price": unit_price,
                "tax": distributed_tax,
                "line_total": "",
            }
        )
    return rows


def normalize_line_items(line_items):
    if not isinstance(line_items, list):
        return []

    normalized = []
    for item in line_items:
        if not isinstance(item, dict):
            continue

        item_id = str(item.get("item_id", "")).strip()
        description = str(item.get("description", "")).strip()
        quantity = str(item.get("quantity", "")).strip()
        unit_price = str(item.get("unit_price", "")).strip()
        tax = str(item.get("tax", "")).strip()
        line_total = str(item.get("line_total", "")).strip()

        discovered_ids = []
        if item_id:
            discovered_ids.extend(extract_item_ids_from_text(item_id))
        # Keep ID extraction strict to ID-bearing fields only.
        # Extracting from description can introduce non-ID tokens (e.g. product words).
        discovered_ids.extend(extract_item_ids_from_text(item.get("notes", "")))
        discovered_ids.extend(extract_item_ids_from_text(item.get("details", "")))

        if discovered_ids and len(discovered_ids) > 1:
            distributed_tax = tax
            parsed_tax = extract_numeric_amount(tax)
            if parsed_tax is not None:
                distributed_tax = format_amount(parsed_tax / len(discovered_ids))

            for discovered_id in discovered_ids:
                normalized.append(
                    {
                        "item_id": discovered_id,
                        "description": description,
                        "quantity": "1",
                        "unit_price": unit_price,
                        "tax": distributed_tax,
                        "line_total": "",
                    }
                )
        else:
            normalized.append(
                {
                    "item_id": item_id,
                    "description": description,
                    "quantity": quantity if quantity else "1",
                    "unit_price": unit_price,
                    "tax": tax,
                    "line_total": line_total,
                }
            )

    return normalized


def infer_total_tax(extracted_json):
    if not isinstance(extracted_json, dict):
        return None

    direct_tax = extract_numeric_amount(extracted_json.get("tax"))

    additional_fields = extracted_json.get("additional_fields", {})
    if not isinstance(additional_fields, dict):
        additional_fields = {}

    cgst = None
    sgst = None
    igst = None
    for key, value in additional_fields.items():
        key_l = str(key).lower()
        amount = extract_numeric_amount(value)
        if amount is None:
            continue
        if "cgst" in key_l:
            cgst = amount
        elif "sgst" in key_l:
            sgst = amount
        elif "igst" in key_l:
            igst = amount

    component_tax = 0.0
    has_components = False
    for amount in (cgst, sgst, igst):
        if amount is not None:
            component_tax += amount
            has_components = True

    if has_components:
        return component_tax
    return direct_tax


def distribute_tax_per_item(line_items, extracted_json):
    if not isinstance(line_items, list) or not line_items:
        return line_items

    total_tax = infer_total_tax(extracted_json)
    if total_tax is None:
        return line_items

    per_item_tax = format_amount(total_tax / len(line_items))
    distributed = []
    for item in line_items:
        row = dict(item) if isinstance(item, dict) else {}
        row["tax"] = per_item_tax
        distributed.append(row)
    return distributed


def infer_total_amount(extracted_json):
    if not isinstance(extracted_json, dict):
        return None

    for key in ("subtotal", "total_amount"):
        amount = extract_numeric_amount(extracted_json.get(key))
        if amount is not None:
            return amount
    return None


def fill_missing_unit_price(line_items, extracted_json):
    if not isinstance(line_items, list) or not line_items:
        return line_items

    discovered_prices = []
    for item in line_items:
        if not isinstance(item, dict):
            continue
        amount = extract_numeric_amount(item.get("unit_price"))
        if amount is not None and amount > 0:
            discovered_prices.append(amount)

    inferred_price = None
    if discovered_prices:
        inferred_price = discovered_prices[0]
    else:
        total_amount = infer_total_amount(extracted_json)
        if total_amount is not None and len(line_items) > 0:
            inferred_price = total_amount / len(line_items)

    if inferred_price is None:
        return line_items

    formatted_price = format_amount(inferred_price)
    updated = []
    for item in line_items:
        row = dict(item) if isinstance(item, dict) else {}
        current_price = extract_numeric_amount(row.get("unit_price"))
        if current_price is None or current_price <= 0:
            row["unit_price"] = formatted_price
        updated.append(row)
    return updated


def add_line_totals(line_items):
    if not isinstance(line_items, list):
        return line_items

    updated = []
    for item in line_items:
        row = dict(item) if isinstance(item, dict) else {}
        qty = extract_numeric_amount(row.get("quantity"))
        price = extract_numeric_amount(row.get("unit_price"))
        if qty is not None and price is not None:
            row["line_total"] = format_amount(qty * price)
        updated.append(row)
    return updated


def ensure_quantity_per_row(line_items):
    if not isinstance(line_items, list):
        return line_items

    updated = []
    has_multiple_rows = len(line_items) > 1
    for item in line_items:
        row = dict(item) if isinstance(item, dict) else {}
        qty = extract_numeric_amount(row.get("quantity"))
        if qty is None or qty <= 0:
            row["quantity"] = "1"
        elif has_multiple_rows and qty > 1:
            # In exploded view, each row represents one item code.
            row["quantity"] = "1"
        updated.append(row)
    return updated


def compute_totals(line_items):
    total_qty = 0.0
    total_amount = 0.0
    total_tax = 0.0

    for item in line_items:
        if not isinstance(item, dict):
            continue
        qty = extract_numeric_amount(item.get("quantity"))
        line_total = extract_numeric_amount(item.get("line_total"))
        unit_price = extract_numeric_amount(item.get("unit_price"))
        tax = extract_numeric_amount(item.get("tax"))

        if qty is not None:
            total_qty += qty

        if line_total is not None:
            total_amount += line_total
        elif qty is not None and unit_price is not None:
            total_amount += qty * unit_price
        if tax is not None:
            total_tax += tax

    return total_qty, total_amount, total_tax


def repair_line_items_if_needed(client, image_bytes_list, extracted_json, normalized_items):
    if not isinstance(extracted_json, dict):
        return normalized_items, None

    original_items = extracted_json.get("line_items", [])
    expected_count = get_expected_item_count(original_items)
    current_count = len(normalized_items)

    if not expected_count or current_count == expected_count:
        return normalized_items, None

    fallback_ids = extract_item_ids_fallback(client, image_bytes_list, expected_count)
    text_parsed_ids = extract_item_ids_from_any_text(extracted_json)

    best_ids = fallback_ids if len(fallback_ids) >= len(text_parsed_ids) else text_parsed_ids
    if len(best_ids) < expected_count:
        return normalized_items, (
            f"Detected {current_count} IDs, expected {expected_count} from quantity. "
            "Fallback ID extraction could not fully recover all entries."
        )
    if len(best_ids) > expected_count:
        best_ids = best_ids[:expected_count]

    template_item = original_items[0] if original_items and isinstance(original_items[0], dict) else {}
    rebuilt = build_rows_from_ids(best_ids, template_item)
    return rebuilt, f"Recovered all {expected_count} IDs using focused transporter parsing."


def render_output_tables(extracted_json, status_note=None, prepared_line_items=None):
    scalar_values = {}
    line_items = prepared_line_items if isinstance(prepared_line_items, list) else []

    for key, value in extracted_json.items():
        if key == "line_items" and isinstance(value, list) and not line_items:
            line_items = normalize_line_items(value)
        elif isinstance(value, (dict, list)):
            scalar_values[key] = json.dumps(value, ensure_ascii=False)
        else:
            scalar_values[key] = "" if value is None else str(value)

    st.subheader("Extracted Fields")
    field_df = pd.DataFrame(
        [{"Field": k.replace("_", " ").title(), "Value": v} for k, v in scalar_values.items()]
    )
    st.dataframe(field_df, use_container_width=True, hide_index=True)

    st.subheader("Line Items")
    if status_note:
        st.info(status_note)
    if line_items:
        st.caption(f"Detected {len(line_items)} item row(s).")
        items_df = pd.DataFrame(line_items)
        st.dataframe(items_df, use_container_width=True, hide_index=True)
        total_qty, total_amount, total_tax = compute_totals(line_items)
        if float(total_qty).is_integer():
            total_qty_display = str(int(total_qty))
        else:
            total_qty_display = format_amount(total_qty)
        total_with_tax = total_amount + total_tax
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Quantity", total_qty_display)
        with col2:
            st.metric("Total Amount", format_amount(total_amount))
        with col3:
            st.metric("Total Tax", format_amount(total_tax))
        with col4:
            st.metric("Grand Total (Incl Tax)", format_amount(total_with_tax))
    else:
        st.caption("No line items detected.")


st.title("Invoice and Document Data Extractor")
st.markdown("Upload an invoice/photo/PDF, extract full data as JSON, and view clean tabular output.")

client = get_client()
if client is None:
    st.warning(
        "Set `GROQ_API_KEY` in environment or create `.env` in project folder. "
        f"Expected path: {DOTENV_PATH}"
    )

uploaded_file = st.file_uploader("Upload invoice or document", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file and client:
    images = prepare_images_from_upload(uploaded_file)

    if st.button("Extract Data", type="primary"):
        with st.spinner("Extracting structured data..."):
            try:
                extracted_json = extract_data_from_images(client, images)
            except Exception as exc:
                st.error(f"Extraction failed: {exc}")
                st.stop()

        st.success("Extraction complete.")
        normalized = normalize_line_items(extracted_json.get("line_items", []))
        repaired_items, status_note = repair_line_items_if_needed(client, images, extracted_json, normalized)
        repaired_items = ensure_quantity_per_row(repaired_items)
        repaired_items = fill_missing_unit_price(repaired_items, extracted_json)
        repaired_items = distribute_tax_per_item(repaired_items, extracted_json)
        repaired_items = add_line_totals(repaired_items)
        total_qty, total_amount, total_tax = compute_totals(repaired_items)
        extracted_json["line_items"] = repaired_items
        extracted_json["line_items_summary"] = {
            "total_rows": len(repaired_items),
            "total_quantity": int(total_qty) if float(total_qty).is_integer() else total_qty,
            "total_amount": format_amount(total_amount),
            "total_tax": format_amount(total_tax),
            "grand_total_including_tax": format_amount(total_amount + total_tax),
        }
        render_output_tables(extracted_json, status_note=status_note, prepared_line_items=repaired_items)

        raw_json = json.dumps(extracted_json, indent=2, ensure_ascii=False)
        with st.expander("Raw JSON Output"):
            st.code(raw_json, language="json")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        download_name = f"extracted_invoice_{timestamp}.json"
        st.download_button(
            "Download JSON",
            data=raw_json,
            file_name=download_name,
            mime="application/json",
        )
