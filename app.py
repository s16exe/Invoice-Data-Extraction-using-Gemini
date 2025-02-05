import os
import base64
import json
import re
import streamlit as st
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError
import google.generativeai as genai
import pandas as pd


class InvoiceItem(BaseModel):
    item_name: str = Field(..., description="Name of the item")
    quantity: float = Field(..., description="Quantity of the item")
    unit_price: float = Field(..., description="Price per unit")
    total_price: float = Field(..., description="Total price for the item")

class InvoiceDetails(BaseModel):
    invoice_number: str = Field(..., description="Invoice number")
    vendor_name: str = Field(..., description="Name of the vendor")
    invoice_date: str = Field(..., description="Date of the invoice")
    total_amount: float = Field(..., description="Total invoice amount")
    tax_amount: float = Field(..., description="Tax amount")
    items: List[InvoiceItem] = Field(..., description="List of invoice items")

class GeminiInvoiceAnalyzer:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def encode_image(self, image_file) -> str:
        return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_invoice(self, image_file) -> InvoiceDetails:
        base64_image = self.encode_image(image_file)
        
        prompt = """
        Carefully extract the following details from the invoice in a strict JSON format:
        {
            "invoice_number": "string",
            "vendor_name": "string",
            "invoice_date": "YYYY-MM-DD",
            "total_amount": 0.00,
            "tax_amount": 0.00,
            "items": [
                {
                    "item_name": "string",
                    "quantity": 0.0,
                    "unit_price": 0.00,
                    "total_price": 0.00
                }
            ]
        }

        Rules:
        - Ensure all fields are filled
        - Use realistic values
        - If a field is not found, use a placeholder or approximate value
        - Provide a valid JSON response
        """
        
        try:
            response = self.model.generate_content([
                prompt, 
                {
                    'mime_type': 'image/jpeg',
                    'data': base64_image
                }
            ])
            
            # Multiple methods to extract JSON
            json_response = self._extract_json(response.text)
            
            # Validate the extracted JSON
            return InvoiceDetails(**json_response)
        
        except ValidationError as ve:
            st.error(f"Validation Error: {ve}")
            # Provide a default or sample response
            return self._get_default_invoice_details()
        
        except Exception as e:
            st.error(f"Error analyzing invoice: {e}")
            return self._get_default_invoice_details()

    def _extract_json(self, text: str) -> dict:
        # Multiple JSON extraction methods
        methods = [
            self._extract_json_between_brackets,
            self._extract_json_with_regex,
            self._extract_json_with_codeblock
        ]
        
        for method in methods:
            try:
                json_dict = method(text)
                if json_dict:
                    return json_dict
            except Exception:
                continue
        
        st.warning("Could not extract JSON from response")
        return {}

    def _extract_json_between_brackets(self, text: str) -> dict:
        # Find JSON between first { and last }
        start = text.find('{')
        end = text.rfind('}') + 1
        
        if start != -1 and end != -1:
            json_str = text[start:end]
            return json.loads(json_str)
        return {}

    def _extract_json_with_regex(self, text: str) -> dict:
        # Use regex to find JSON
        json_match = re.search(r'\{.*?\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return {}

    def _extract_json_with_codeblock(self, text: str) -> dict:
        # Extract JSON from markdown code block
        code_block_match = re.search(r'```json\n(.*?)```', text, re.DOTALL)
        if code_block_match:
            return json.loads(code_block_match.group(1))
        return {}

    def _get_default_invoice_details(self) -> InvoiceDetails:
        # Provide a default/sample invoice details
        return InvoiceDetails(
            invoice_number="N/A",
            vendor_name="Unknown Vendor",
            invoice_date="2023-01-01",
            total_amount=0.00,
            tax_amount=0.00,
            items=[
                InvoiceItem(
                    item_name="Sample Item",
                    quantity=0.0,
                    unit_price=0.00,
                    total_price=0.00
                )
            ]
        )

# Streamlit UI
def main():
    st.title("Invoice Analyzer")
    
    # Set your Google AI Studio API Key
    API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyB1jmZxMDoB8AdkRLqysPTzmwRPSCtMGFk')
    
    # Initialize the Invoice Analyzer
    invoice_analyzer = GeminiInvoiceAnalyzer(API_KEY)
    
    # File uploader for invoice image
    uploaded_file = st.file_uploader("Upload Invoice Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Invoice', use_column_width=True)
        
        if st.button("Analyze Invoice"):
            try:
                # Analyze Invoice
                invoice_details = invoice_analyzer.analyze_invoice(uploaded_file)
                
                # Display Structured Invoice Details
                st.subheader("Invoice Analysis Results:")
                
                # Create columns for better display
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Invoice Number:** {invoice_details.invoice_number}")
                    st.write(f"**Vendor Name:** {invoice_details.vendor_name}")
                    st.write(f"**Invoice Date:** {invoice_details.invoice_date}")
                
                with col2:
                    st.write(f"**Total Amount:** ${invoice_details.total_amount:.2f}")
                    st.write(f"**Tax Amount:** ${invoice_details.tax_amount:.2f}")
                
                # Display Items
                st.subheader("Invoice Items")
                items_df = pd.DataFrame([item.dict() for item in invoice_details.items])
                st.dataframe(items_df)
                
                # Full JSON view
                with st.expander("Full JSON Response"):
                    st.json(invoice_details.dict())
            
            except Exception as e:
                st.error(f"Invoice analysis failed: {e}")

if __name__ == "__main__":
    main()