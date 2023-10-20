from pathlib import Path
from typing import Optional, Union

from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate)
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from pydantic.v1 import BaseModel, Field

from create_summaries import instantiate_model, load_template

SAVE_DIR = Path("data", "summary_comparison")
SUMMARIES_DIR = Path("data", "summaries")


class DocumentInput(BaseModel):
    question: str = Field()


def document_comparison_tool(
    model_name: str,
    sections_2015_fname: Union[Path, str],
    sections_2023_fname: Union[Path, str],
    document_comparison_prompt: str,
    save_dir: Optional[Union[Path, str]] = None,
) -> str:
    """Langchain document compariosn as implemented here:
    https://python.langchain.com/docs/integrations/toolkits/document_comparison_toolkit

    Args:
        model_name (str): model to use, should be one of "gpt-3.5-turbo-0613",
        "text-davinci-003", or "gpt-4"
        sections_2015_fname (Union[Path, str]): filepath of 2015 "sections" summary
        sections_2023_fname (Union[Path, str]): filepath of 2023 "sections" summary
        document_comparison_prompt (str): prompt to use for document comparison
        save_dir (Optional[Union[Path, str]], optional): directory to save comparison text. Defaults to None.

    Returns:
        str: comparison of summaries
    """
    llm = instantiate_model(model_name)

    tools = []
    files = [
        {
            "name": "2015-TandCS",
            "path": str(sections_2015_fname),
        },
        {
            "name": "2023-TandCS",
            "path": str(sections_2023_fname),
        },
    ]

    for file in files:
        loader = TextLoader(file["path"])
        pages = loader.load_and_split()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(pages)
        embeddings = OpenAIEmbeddings()
        retriever = FAISS.from_documents(docs, embeddings).as_retriever()

        # Wrap retrievers in a Tool
        tools.append(
            Tool(
                args_schema=DocumentInput,
                name=file["name"],
                description=f"useful when you want to answer questions about {file['name']}",
                func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever),
            )
        )
    agent = initialize_agent(
        agent=AgentType.OPENAI_FUNCTIONS,
        tools=tools,
        llm=llm,
        verbose=True,
    )
    document_comparison_prompt = (
        "What are the key differences between the 2015-TandCS and 2023-TandCS?"
    )
    output = agent({"input": document_comparison_prompt})

    if save_dir is not None:
        with open(save_dir, "w", encoding="utf-8") as f:
            f.write(output["output"])
        return output["output"]
    return output["output"]


def comparison_prompt_engineering(
    model_name: str,
    sections_2015_fname: Union[Path, str],
    sections_2023_fname: Union[Path, str],
    query: str,
    save_dir: Optional[Union[Path, str]] = None,
):
    """
    Compare two versions of documents using an AI model and retrieve differences.

    Args:
        model_name (str): The name of the language model to use for comparison.
        sections_2015_fname (Union[Path, str]): The path or filename of the 2015 document.
        sections_2023_fname (Union[Path, str]): The path or filename of the 2023 document.
        query (str): The comparison query to answer.
        save_dir (Optional[Union[Path, str]]): Optional. If provided, the comparison result will be saved to this file.

    Returns:
        str: The comparison result as a text string.

    The function loads two documents, splits them into smaller chunks, computes embeddings,
    and uses a language model to answer the comparison query. It can save the comparison
    result to a file if the 'save_dir' parameter is provided.
    """

    llm = instantiate_model(model_name)
    with open(sections_2015_fname, "r", encoding="utf-8") as f:
        section_2015 = f.read()
    with open(sections_2023_fname, "r", encoding="utf-8") as f:
        section_2023 = f.read()

    prompt_template = PromptTemplate(input_variables=["text1", "text2"], template=query)
    prompt_template_formatted = prompt_template.format(
        text1=section_2015, text2=section_2023
    )

    if model_name != "text-davinci-003":
        human_message_prompt = HumanMessagePromptTemplate.from_template(query)
        chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
        gpt_prompt_engineering_output = llm(
            chat_prompt.format_prompt(
                text1=section_2015, text2=section_2023
            ).to_messages()
        )
        prompt_engineering_output = gpt_prompt_engineering_output.content
    else:
        prompt_engineering_output = llm(prompt_template_formatted)
    if save_dir is not None:
        with open(save_dir, "w", encoding="utf-8") as f:
            f.write(prompt_engineering_output)
        return prompt_engineering_output
    return prompt_engineering_output


def main():
    load_dotenv()
    sections_2015_fname = Path(SUMMARIES_DIR, "2015_sections_summary.txt")
    sections_2023_fname = Path(SUMMARIES_DIR, "2023_sections_summary.txt")

    # Document comparison
    document_comparison_save_dir = Path(SAVE_DIR, "document_comparison_output.txt")
    query = load_template(Path("prompts_templates", "document_comparison_prompt.txt"))

    if not document_comparison_save_dir.exists():
        document_comparison_tool(
            model_name="gpt-3.5-turbo-0613",
            sections_2015_fname=sections_2015_fname,
            sections_2023_fname=sections_2023_fname,
            document_comparison_prompt=query,
            save_dir=document_comparison_save_dir,
        )

    # prompt engineering - davinci
    prompt_engineering_davinci_save_dir = Path(
        SAVE_DIR, "prompt_engineering_output_davinci.txt"
    )
    query = load_template(
        Path("prompts_templates", "prompt_engineering_comparison_prompt.txt")
    )
    if not prompt_engineering_davinci_save_dir.exists():
        comparison_prompt_engineering(
            model_name="text-davinci-003",
            sections_2015_fname=sections_2015_fname,
            sections_2023_fname=sections_2023_fname,
            query=query,
            save_dir=prompt_engineering_davinci_save_dir,
        )

    # prompt engineering - gpt
    prompt_engineering_gpt_save_dir = Path(
        SAVE_DIR, "prompt_engineering_output_gpt.txt"
    )
    query = load_template(
        Path("prompts_templates", "prompt_engineering_comparison_prompt.txt")
    )
    if not prompt_engineering_gpt_save_dir.exists():
        comparison_prompt_engineering(
            model_name="gpt-3.5-turbo-0613",
            sections_2015_fname=sections_2015_fname,
            sections_2023_fname=sections_2023_fname,
            query=query,
            save_dir=prompt_engineering_gpt_save_dir,
        )


if __name__ == "__main__":
    main()
