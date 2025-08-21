import streamlit as st
import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from transformers.utils import logging

# Set logging level to reduce verbosity
logging.set_verbosity_error()

# Set page config
st.set_page_config(page_title="CodeT5 Code Generator", page_icon="🐍")

# App title
st.title("🐍 CodeT5 Code Generator")
st.write("Generate Python code using a fine-tuned CodeT5 model")

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None

# Function to fix Python indentation
def fix_python_indentation(code):
    lines = code.split('\n')
    fixed_lines = []
    indent_level = 0
    in_multiline_string = False
    string_delimiter = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            fixed_lines.append('')
            continue
            
        # Handle multiline strings
        if in_multiline_string:
            fixed_lines.append(('    ' * indent_level) + line)
            if string_delimiter in line:
                in_multiline_string = False
            continue
        elif any(stripped.startswith(delimiter) for delimiter in ['"""', "'''", 'r"""', "r'''"]):
            in_multiline_string = True
            string_delimiter = stripped[:3] if stripped.startswith('r') else stripped[:3]
            fixed_lines.append(('    ' * indent_level) + line)
            continue

        # Handle indentation for code blocks
        if stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'with ')):
            fixed_line = ('    ' * indent_level) + stripped
            if stripped.endswith(':') and not stripped.startswith(('"""', "'''")):
                indent_level += 1
        elif stripped.startswith(('elif ', 'else:', 'except ', 'finally:')):
            indent_level = max(0, indent_level - 1)
            fixed_line = ('    ' * indent_level) + stripped
            if stripped.endswith(':') and not stripped.startswith(('"""', "'''")):
                indent_level += 1
        elif stripped.startswith(('return ', 'break', 'continue', 'pass', 'raise ', 'yield ')):
            fixed_line = ('    ' * indent_level) + stripped
        else:
            fixed_line = ('    ' * indent_level) + stripped

        fixed_lines.append(fixed_line)

    return '\n'.join(fixed_lines)

# Sidebar for settings
with st.sidebar:
    st.header("Settings")

    model_path = st.text_input(
        "Model Path",
        value="model",
        help="Path to your fine-tuned model"
    )

    max_length = st.slider("Max Length", 100, 1000, 250, 50)
    num_beams = st.slider("Number of Beams", 1, 10, 5, 1)
    temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)

    if st.button("Load Model"):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            st.write(f"Using device: {device}")
            
            # Clear any previous model from memory
            if st.session_state.model is not None:
                del st.session_state.model
                torch.cuda.empty_cache()
            
            # Load tokenizer and model
            with st.spinner("Loading tokenizer..."):
                tokenizer = RobertaTokenizer.from_pretrained(model_path)
            
            with st.spinner("Loading model..."):
                model = T5ForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    use_safetensors=True,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                
                if not torch.cuda.is_available():
                    model = model.to(device)
                
                model.eval()

            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.device = device
            st.session_state.model_loaded = True
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.info("""
            Common issues:
            1. Make sure the model path is correct
            2. Check if all model files are present
            3. Try restarting the application
            """)

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

if st.button("Generate Code"):
    if not st.session_state.model_loaded:
        st.error("Please load the model first using the sidebar options")
    elif not instruction:
        st.warning("Please provide an instruction")
    else:
        # Format the prompt
        if input_data:
            prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_data}

### Response:
"""
        else:
            prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

        # Generate code
        with st.spinner("Generating code..."):
            try:
                inputs = st.session_state.tokenizer(
                    prompt,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(st.session_state.device)

                with torch.no_grad():
                    outputs = st.session_state.model.generate(
                        **inputs,
                        max_length=max_length,
                        num_beams=num_beams,
                        temperature=temperature,
                        early_stopping=True,
                        do_sample=True
                    )

                predicted_code = st.session_state.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )

                # Extract only the code after "### Response:"
                if "### Response:" in predicted_code:
                    predicted_code = predicted_code.split("### Response:")[1].strip()

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
