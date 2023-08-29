# flake8: noqa
from langchain.prompts.prompt import PromptTemplate

_template = """Dada a seguinte conversa e uma pergunta de acompanhamento, reformule a pergunta de acompanhamento para ser uma pergunta independente.

Histórico do chat: {chat_history}
Entrada de acompanhamento: {question}
Pergunta independente:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

prompt_template = """Utilize as seguintes informações de contexto para responder a pergunta no final. Se você não souber a resposta, apenas diga que não sabe, não tente inventar uma resposta.

{context}

Pergunta: {question}
Resposta útil:
"""
QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
