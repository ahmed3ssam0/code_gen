import streamlit as st
import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import re
import os

# Set page config
st.set_page_config(page_title="CodeT5 Code Generator", page_icon="🐍")

# App title
st.title("🐍 CodeT5 Code Generator")
st.write("Generate Python code using a fine-tuned CodeT5 model")

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False


# Function to fix Python indentation
def fix_python_indentation(code):
    lines = code.split('\n')
    fixed_lines = []
    indent_level = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            fixed_lines.append("")  # keep empty lines
            continue

        # Handle block starters
        if stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'else:', 'elif ')):
            fixed_line = ('    ' * indent_level) + stripped
            fixed_lines.append(fixed_line)
            if stripped.endswith(':'):
                indent_level += 1
        elif stripped.startswith(('return', 'break', 'continue', 'pass')):
            fixed_line = ('    ' * indent_level) + stripped
            fixed_lines.append(fixed_line)
        else:
            fixed_line = ('    ' * indent_level) + stripped
            fixed_lines.append(fixed_line)

            # If line ends with ":" but wasn't caught above (e.g. try:, except:)
            if stripped.endswith(':'):
                indent_level += 1

        # Adjust indent after return/break/etc.
        if stripped in ('return', 'break', 'continue', 'pass'):
            indent_level = max(0, indent_level - 1)

    return '\n'.join(fixed_lines)


# Sidebar for settings
with st.sidebar:
    st.header("Settings")

    # Fixed model path
    model_path = "model"

    max_length = st.slider("Max Length", 100, 1000, 250, 50)
    num_beams = st.slider("Number of Beams", 1, 10, 5, 1)

    # Auto load model
    if not st.session_state.model_loaded:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
    
            # Load model with safe handling for meta tensors
            model = T5ForConditionalGeneration.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float32
            )
    
            # If not using CUDA, move to CPU manually
            if device.type == "cpu":
                model = model.to("cpu")
    
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.device = device
            st.session_state.model_loaded = True
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")

# Main input section
st.header("Describe Your Code")

instruction = st.text_area(
    "Instruction",
    height=100,
    placeholder="Describe what you want the code to do"
)

input_data = st.text_input(
    "Input (Optional)",
    placeholder="Optional input data"
)

if st.button("Generate Code") and st.session_state.model_loaded:
    if not instruction:
        st.warning("Please provide an instruction")
    else:
        # Format the prompt
        if input_data:
            prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_data}

### Output:
"""
        else:
            prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Output:
"""

        # Generate code
        with st.spinner("Generating code..."):
            try:
                inputs = st.session_state.tokenizer(
                    prompt,
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                ).to(st.session_state.device)

                outputs = st.session_state.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True
                )

                predicted_code = st.session_state.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )

                # Fix indentation
                fixed_code = fix_python_indentation(predicted_code)

                # Display results
                st.subheader("Generated Code")
                st.code(fixed_code, language='python')

                # Download button
                st.download_button(
                    "Download Code",
                    fixed_code,
                    file_name="generated_code.py",
                    mime="text/x-python"
                )

            except Exception as e:
                st.error(f"Error generating code: {str(e)}")
elif not st.session_state.model_loaded:
    st.info("Please wait, model is loading...")
