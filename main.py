"""Main entrypoint for the app."""
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from langchain.vectorstores import VectorStore
from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings import OpenAIEmbeddings


from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from query_data import get_chain
from schemas import ChatResponse

app = FastAPI()
templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None
COLLECTION_NAME = "carolina"
CONNECTION_STRING = f"postgresql://{os.environ.get('DATABASE_USER')}:{os.environ.get('DATABASE_PASSWORD')}@{os.environ.get('DATABASE_HOST')}:5432/{os.environ.get('DATABASE_NAME')}"
embeddings_service = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_TOKEN"])
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def startup_event():
    # Load the vectorstore from pgvector
    global vectorstore
    vectorstore = PGVector(
        connection_string=CONNECTION_STRING,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings_service
    )
    logger.info("Loaded vectorstore")


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    qa_chain = get_chain(vectorstore, question_handler, stream_handler)
    # Use the below line instead of the above line to enable tracing
    # Ensure `langchain-server` is running
    # qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)

    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            result = await qa_chain.acall(
                {"question": question, "chat_history": chat_history}
            )

            logger.info(f"Question: {question}")
            logger.info(f"Answer: {result}")
            chat_history.append((question, result["answer"]))

            end_resp = ChatResponse(sender="bot", message="", type="end")
            # await websocket.send_json(end_resp.dict())

            # start_resp = ChatResponse(sender="bot", message="", type="start")
            # await websocket.send_json(start_resp.dict())
            if source_docs := result["source_documents"]:
                msg = ChatResponse(sender="bot", message="<br><br>Link para documentação: ", type="stream")
                await websocket.send_json(msg.dict())
                # get all unique url from the property metadata.mdmurl in all documents
                urls = {doc.metadata["mdmurl"] for doc in source_docs}
                for url in urls:
                    url = f"<a target='_blank' href='{url}'>{url}</a>"
                    msg = ChatResponse(sender="bot", message=url, type="stream")
                    await websocket.send_json(msg.dict())
            await websocket.send_json(end_resp.dict())

        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
