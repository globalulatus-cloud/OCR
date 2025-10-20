"""
PDF Text Extractor with OCR Support
Extracts text from both editable and scanned PDFs using EasyOCR
"""

import streamlit as st
import easyocr
import pdfplumber
from pdf2image import convert_from_bytes
from PIL import Image
import numpy as np
import cv2
import io
import tempfile
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="PDF Text Extractor",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = None

@st.cache_resource
def load_ocr_reader():
    """Load EasyOCR reader with multi-language support"""
    # Languages: English, Chinese, Japanese, Korean, Hindi, Bengali, Thai, French, German, Spanish
    return easyocr.Reader([
        'en', 'ch_sim', 'ja', 'ko', 'hi', 'bn', 'th', 
        'fr', 'de', 'es', 'ru', 'ar', 'pt', 'it', 'nl'
    ], gpu=False)

def extract_text_from_page(pdf_page):
    """Extract text from a single PDF page using pdfplumber"""
    try:
        text = pdf_page.extract_text()
        return text if text else ""
    except:
        return ""

def page_to_image(pdf_page_bytes, dpi=200):
    """Convert PDF page bytes to PIL Image"""
    images = convert_from_bytes(pdf_page_bytes, dpi=dpi)
    return images[0] if images else None

def perform_ocr(image, reader):
    """
    Perform OCR on an image and return text with confidence scores
    Returns: (extracted_text, average_confidence, bounding_boxes)
    """
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Perform OCR
    results = reader.readtext(img_array)
    
    if not results:
        return "", 0.0, []
    
    # Extract text and confidence scores
    texts = []
    confidences = []
    bboxes = []
    
    for (bbox, text, conf) in results:
        texts.append(text)
        confidences.append(conf)
        bboxes.append(bbox)
    
    # Calculate average confidence
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    # Join all text with spaces
    full_text = " ".join(texts)
    
    return full_text, avg_confidence, bboxes

def draw_bboxes_on_image(image, bboxes):
    """Draw bounding boxes on image for visualization"""
    img_array = np.array(image.copy())
    
    for bbox in bboxes:
        # Convert bbox to integer coordinates
        pts = np.array(bbox, dtype=np.int32)
        
        # Draw rectangle
        cv2.polylines(img_array, [pts], True, (0, 255, 0), 2)
    
    return Image.fromarray(img_array)

def count_words_and_chars(text):
    """Count words and characters in text"""
    words = len(text.split())
    chars_with_spaces = len(text)
    chars_without_spaces = len(text.replace(" ", "").replace("\n", ""))
    
    return words, chars_with_spaces, chars_without_spaces

def process_pdf(pdf_file, reader, progress_bar, status_text):
    """
    Main function to process PDF and extract all text
    Returns: dictionary with results
    """
    results = {
        'pages': [],
        'total_words': 0,
        'total_chars_with_spaces': 0,
        'total_chars_without_spaces': 0,
        'total_pages': 0,
        'editable_pages': 0,
        'scanned_pages': 0,
        'hybrid_pages': 0
    }
    
    all_text_content = []
    
    try:
        # Open PDF with pdfplumber
        with pdfplumber.open(pdf_file) as pdf:
            total_pages = len(pdf.pages)
            results['total_pages'] = total_pages
            
            for page_num, page in enumerate(pdf.pages, 1):
                status_text.text(f"Processing page {page_num}/{total_pages}...")
                
                # Try to extract editable text first
                editable_text = extract_text_from_page(page)
                
                page_info = {
                    'page_num': page_num,
                    'type': '',
                    'confidence': 'N/A',
                    'text': '',
                    'word_count': 0,
                    'preview_image': None
                }
                
                # Determine if page needs OCR (less than 50 characters means likely scanned)
                if len(editable_text.strip()) < 50:
                    # Convert page to image for OCR
                    pdf_bytes = pdf_file.read()
                    pdf_file.seek(0)
                    
                    # Get single page
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        with pdfplumber.open(pdf_file) as temp_pdf:
                            # This is a workaround - convert specific page
                            pass
                    
                    # For simplicity, convert entire PDF and get the page
                    images = convert_from_bytes(pdf_bytes, first_page=page_num, last_page=page_num, dpi=200)
                    
                    if images:
                        page_image = images[0]
                        
                        # Perform OCR
                        ocr_text, confidence, bboxes = perform_ocr(page_image, reader)
                        
                        if len(editable_text.strip()) > 0:
                            # Hybrid page
                            page_info['type'] = 'Hybrid'
                            page_info['text'] = editable_text + "\n" + ocr_text
                            results['hybrid_pages'] += 1
                        else:
                            # Scanned page
                            page_info['type'] = 'Scanned'
                            page_info['text'] = ocr_text
                            results['scanned_pages'] += 1
                        
                        page_info['confidence'] = f"{confidence * 100:.1f}%"
                        
                        # Draw bounding boxes for preview
                        preview_image = draw_bboxes_on_image(page_image, bboxes)
                        page_info['preview_image'] = preview_image
                else:
                    # Editable page
                    page_info['type'] = 'Editable'
                    page_info['text'] = editable_text
                    results['editable_pages'] += 1
                
                # Count words and characters
                words, chars_with, chars_without = count_words_and_chars(page_info['text'])
                page_info['word_count'] = words
                
                results['total_words'] += words
                results['total_chars_with_spaces'] += chars_with
                results['total_chars_without_spaces'] += chars_without
                
                results['pages'].append(page_info)
                
                # Prepare text for output file
                all_text_content.append(
                    f"\nPAGE {page_num} | Type: {page_info['type']} | Confidence: {page_info['confidence']}\n"
                    f"{'-' * 60}\n"
                    f"{page_info['text']}\n"
                )
                
                # Update progress
                progress_bar.progress(page_num / total_pages)
        
        # Create final text output
        header = (
            "=" * 60 + "\n"
            "PDF TEXT EXTRACTION REPORT\n"
            "=" * 60 + "\n"
            f"Total Word Count: {results['total_words']}\n"
            f"Total Character Count (with spaces): {results['total_chars_with_spaces']}\n"
            f"Total Character Count (without spaces): {results['total_chars_without_spaces']}\n"
            f"Total Pages Processed: {results['total_pages']}\n"
            f"Editable Pages: {results['editable_pages']}\n"
            f"Scanned Pages: {results['scanned_pages']}\n"
            f"Hybrid Pages: {results['hybrid_pages']}\n"
            "=" * 60 + "\n"
        )
        
        results['extracted_text'] = header + "".join(all_text_content)
        
        return results
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

# Main App UI
def main():
    st.title("📄 PDF Text Extractor with OCR")
    st.markdown("Extract text from both **editable** and **scanned** PDFs with multi-language support")
    
    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(
            """
            **Features:**
            - Multi-language OCR (80+ languages)
            - Handles editable, scanned, and hybrid PDFs
            - Confidence scores for OCR results
            - Visual bounding box preview
            - Detailed statistics
            
            **Max file size:** 100 MB
            """
        )
        
        st.header("📊 Supported Languages")
        st.markdown(
            """
            - English, Chinese, Japanese, Korean
            - Hindi, Bengali, Thai, Arabic
            - French, German, Spanish, Portuguese
            - Russian, Italian, Dutch
            - And many more...
            """
        )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF file (max 100MB)"
    )
    
    if uploaded_file is not None:
        # Check file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        
        if file_size_mb > 100:
            st.error("❌ File size exceeds 100MB limit. Please upload a smaller file.")
            return
        
        # Display file info
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.success(f"📁 **File:** {uploaded_file.name}")
        with col2:
            st.info(f"📦 **Size:** {file_size_mb:.2f} MB")
        with col3:
            process_button = st.button("🚀 Extract Text", type="primary")
        
        # Process PDF
        if process_button:
            with st.spinner("Loading OCR engine..."):
                reader = load_ocr_reader()
            
            st.info("⏳ Processing PDF... This may take a few minutes for large files.")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process the PDF
            results = process_pdf(uploaded_file, reader, progress_bar, status_text)
            
            if results:
                st.session_state.processed_results = results
                status_text.text("✅ Processing complete!")
                st.balloons()
        
        # Display results if available
        if st.session_state.processed_results:
            results = st.session_state.processed_results
            
            st.markdown("---")
            st.header("📊 Extraction Results")
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Pages", results['total_pages'])
            with col2:
                st.metric("Total Words", results['total_words'])
            with col3:
                st.metric("Chars (with spaces)", results['total_chars_with_spaces'])
            with col4:
                st.metric("Chars (no spaces)", results['total_chars_without_spaces'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Editable Pages", results['editable_pages'], delta_color="normal")
            with col2:
                st.metric("Scanned Pages", results['scanned_pages'], delta_color="inverse")
            with col3:
                st.metric("Hybrid Pages", results['hybrid_pages'], delta_color="off")
            
            # Download button
            st.download_button(
                label="📥 Download Extracted Text (.txt)",
                data=results['extracted_text'],
                file_name=f"{Path(uploaded_file.name).stem}_extracted.txt",
                mime="text/plain",
                type="primary"
            )
            
            # Page details
            st.markdown("---")
            st.subheader("📄 Page Details")
            
            # Create tabs for different views
            tab1, tab2 = st.tabs(["📋 Table View", "🖼️ Preview with Bounding Boxes"])
            
            with tab1:
                # Display page information in a table
                page_data = []
                for page in results['pages']:
                    page_data.append({
                        'Page': page['page_num'],
                        'Type': page['type'],
                        'Confidence': page['confidence'],
                        'Word Count': page['word_count']
                    })
                
                st.dataframe(page_data, use_container_width=True)
            
            with tab2:
                # Show pages with bounding boxes
                st.info("Preview of detected text regions (only for scanned/hybrid pages)")
                
                for page in results['pages']:
                    if page['preview_image']:
                        with st.expander(f"Page {page['page_num']} - {page['type']} (Confidence: {page['confidence']})"):
                            st.image(page['preview_image'], use_container_width=True)

if __name__ == "__main__":
    main()
