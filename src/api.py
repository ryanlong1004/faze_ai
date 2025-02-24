import tempfile

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Air Speech To Text API",
    version="1.0",
    contact={
        "name": "Ryan Long",
        "email": "rlong@wasabi.com",
    },
    docs_url="/",
)


@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    """
    Transcribe the uploaded audio file.

    Args:
        file (UploadFile): The uploaded audio file.
        language (str): The language of the audio.
        translate (bool): Whether to translate the audio.

    Returns:
        JSONResponse: The transcriptions of the audio.
    """
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        temp_file.write(contents)
        temp_file.flush()  # Ensure data is written to disk
        # result = fx(temp_file.name)
        return JSONResponse(content=None)
