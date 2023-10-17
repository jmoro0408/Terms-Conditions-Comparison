from pathlib import Path
from typing import Optional, Union

from dotenv import load_dotenv
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.document_loaders import Docx2txtLoader
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

SUMMARY_SAVE_DIR = Path("data", "summaries")
DATA_DIR = Path("data", "raw")


def load_document(document_fname: Union[Path, str]) -> list[Document]:
    return Docx2txtLoader(str(document_fname)).load()  # str reqd for loader


def load_template(template_fname: Union[Path, str]) -> str:
    with open(template_fname, "r") as f:
        template = f.read()
    return template


def instantiate_model(model_name: str, **kwargs):
    if model_name == "gpt-4":
        return ChatOpenAI(model_name="gpt-4", **kwargs)
    if model_name == "text-davinci-003":
        return OpenAI(model="text-davinci-003", **kwargs)


def map_reduce(
    model_name: str, 
    text_split, 
    save_dir: Optional[Union[Path, str]] = None, 
    **kwargs
):
    model = instantiate_model(model_name, **kwargs)
    map_template_fname = Path("prompts_templates", "map_template.txt")
    map_template = load_template(map_template_fname)
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=model, prompt=map_prompt)

    reduce_template_fname = Path("prompts_templates", "reduce_template.txt")
    reduce_template = load_template(reduce_template_fname)
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=model, prompt=reduce_prompt)
    # Combines and iteravely reduces the mapped documents
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="doc_summaries"
    )

    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=4000,
    )

    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )
    output = map_reduce_chain.run(text_split)
    if save_dir is not None:
        with open(save_dir, "w") as f:
            f.write(output)
        return output
    return output


def tokenize_document(
    input_document: list[Document], chunk_size: int, chunk_overlap: int
):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(input_document)


def main():
    load_dotenv()
    toc_2015_fname = Path(DATA_DIR, "Jan 2015.docx")
    toc_2023_fname = Path(DATA_DIR, "Mar 2023.docx")

    data_2015 = load_document(toc_2015_fname)
    data_2023 = load_document(toc_2023_fname)

    # GPT4 map reduce
    split_2015_gpt4 = tokenize_document(data_2015, chunk_size=4000, chunk_overlap=0)
    split_2023_gpt4 = tokenize_document(data_2023, chunk_size=4000, chunk_overlap=0)
    gpt4_2015_mp_fname = Path(SUMMARY_SAVE_DIR, "gpt4_map_reduce_summarized_2015.txt")
    gpt4_2023_mp_fname = Path(SUMMARY_SAVE_DIR, "gpt4_map_reduce_summarized_2023.txt")
    if not gpt4_2015_mp_fname.exists():
        map_reduce(
            "gpt-4",
            split_2015_gpt4,
            save_dir=gpt4_2015_mp_fname, 
            temperature = 0
        )
    if not gpt4_2023_mp_fname.exists():
        map_reduce(
            "gpt-4",
            split_2023_gpt4,
            save_dir=gpt4_2015_mp_fname,
            temperature = 0
        )

    ## Davinci
    split_2015_davinci = tokenize_document(data_2015, chunk_size=1000, chunk_overlap=0)
    split_2023_davinci = tokenize_document(data_2023, chunk_size=1000, chunk_overlap=0)
    davinci_2015_mp_fname = Path(SUMMARY_SAVE_DIR, "davinci_map_reduce_summarized_2015.txt")
    davinci_2023_mp_fname = Path(SUMMARY_SAVE_DIR, "davinci_map_reduce_summarized_2023.txt")
    if not davinci_2015_mp_fname.exists():
        map_reduce(
            "text-davinci-003",
            split_2015_davinci,
            save_dir=davinci_2015_mp_fname, 
            temperature = 0, 
            max_tokens=1000
        )

    if not davinci_2023_mp_fname.exists():
        map_reduce(
            "text-davinci-003",
            split_2023_davinci,
            save_dir=davinci_2023_mp_fname,
            temperature = 0, 
            max_tokens=1000
        )


if __name__ == "__main__":
    main()