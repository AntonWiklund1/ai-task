# AI Test

## Setup Instructions

### 1. Create a Python Virtual Environment

```bash
python -m venv venv
```

### 2. Activate the Virtual Environment

- **macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```
- **Windows (Git Bash or WSL):**
  ```bash
  source venv/Scripts/activate
  ```
- **Windows (Command Prompt or PowerShell):**
  ```bash
  .\venv\Scripts\activate
  ```

### 3. Install Dependencies

After activating the virtual environment, install the required packages:

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root of your project and add the following variables:

```env
POSTGRESQL_URL=your_postgresql_url
VOYAGE_API_KEY=your_voyage_api_key
PINECONE_HOST=your_pinecone_host
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_pinecone_index_name
GEMINI_API_KEY=your_gemini_api_key
```

Replace `your_...` with your actual credentials and values.

### 5. Run the API

```bash
python main.py
```

### 6. Test the Application

You can now test the application using the `
