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
    model_name: str, text_split, save_dir: Optional[Union[Path, str]] = None, **kwargs
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


def string_between_substrings(input_str: str, start: str, end: Optional[str]):
    if end is None:
        return (input_str.split(start))[1]
    return (input_str.split(start))[1].split(end)[0]


def map_reduce_sections_2015(
    document_2015: list[Document], save_dir: Optional[Union[Path, str]] = None
):
    # TODO Lot of repeated code between here and 2023, can be abstracted out
    ## Summary by section
    delimiters_2015 = [
        "A. ITUNES STORE, MAC APP STORE, APP STORE AND IBOOKS STORE TERMS OF SALE",
        "B. ITUNES STORE TERMS AND CONDITIONS",
        "C. MAC APP STORE, APP STORE AND IBOOKS STORE TERMS AND CONDITIONS",
    ]
    raw_text_2015 = document_2015[0].page_content
    raw_text_2015 = raw_text_2015[
        200:
    ]  # removing the initial text as this matches the delimiters
    a_section = string_between_substrings(
        raw_text_2015, delimiters_2015[0], delimiters_2015[1]
    )

    b_section = string_between_substrings(
        raw_text_2015, delimiters_2015[1], delimiters_2015[2]
    )

    c_section = string_between_substrings(raw_text_2015, delimiters_2015[2], None)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=4000, chunk_overlap=0
    )
    a_doc_2015 = text_splitter.create_documents([a_section])
    b_doc_2015 = text_splitter.create_documents([b_section])
    c_doc_2015 = text_splitter.create_documents([c_section])

    a_split_2015 = text_splitter.split_documents(a_doc_2015)
    b_split_2015 = text_splitter.split_documents(b_doc_2015)
    c_split_2015 = text_splitter.split_documents(c_doc_2015)

    output_map_reduce_15_a = map_reduce("gpt-4", a_split_2015, temperature=0)

    output_map_reduce_15_b = map_reduce("gpt-4", b_split_2015, temperature=0)

    output_map_reduce_15_c = map_reduce("gpt-4", c_split_2015, temperature=0)
    sections_2015_summary = (
        output_map_reduce_15_a
        + "\n"
        + output_map_reduce_15_b
        + "\n"
        + output_map_reduce_15_c
    )
    if save_dir is not None:
        with open(save_dir, "w") as f:
            f.write(sections_2015_summary)
        return sections_2015_summary
    return sections_2015_summary


def map_reduce_sections_2023(
    document_2023: list[Document], save_dir: Optional[Union[Path, str]] = None
):
    raw_text_2023 = document_2023[0].page_content
    delimiters_2023 = [
        "A. INTRODUCTION TO OUR SERVICES",
        # "B. USING OUR SERVICES",
        "C. YOUR SUBMISSIONS TO OUR SERVICES",
        # "D. FAMILY SHARING",
        # "E. SERIES PASS AND MULTI-PASS",
        # "F. ADDITIONAL APP STORE TERMS (EXCLUDING APPLE ARCADE APPS)",
        "G. ADDITIONAL TERMS FOR CONTENT ACQUIRED FROM THIRD PARTIES",
        # "H. ADDITIONAL APPLE MUSIC TERMS",
        # "I. ADDITIONAL APPLE FITNESS+ TERMS",
        # "J. CARRIER MEMBERSHIP",
        # "K. MISCELLANEOUS TERMS APPLICABLE TO ALL SERVICES",
    ]
    # I've manually split these into lengths of around 2500 - 3000 words

    a_section_23 = string_between_substrings(
        raw_text_2023, delimiters_2023[0], delimiters_2023[1]
    )

    b_section_23 = string_between_substrings(
        raw_text_2023, delimiters_2023[1], delimiters_2023[2]
    )

    c_section_23 = string_between_substrings(raw_text_2023, delimiters_2023[2], None)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=4000, chunk_overlap=0
    )
    a_doc_2023 = text_splitter.create_documents([a_section_23])
    b_doc_2023 = text_splitter.create_documents([b_section_23])
    c_doc_2023 = text_splitter.create_documents([c_section_23])

    a_split_2023 = text_splitter.split_documents(a_doc_2023)
    b_split_2023 = text_splitter.split_documents(b_doc_2023)
    c_split_2023 = text_splitter.split_documents(c_doc_2023)
    output_map_reduce_23_a = map_reduce("gpt-4", a_split_2023, temperature=0)
    output_map_reduce_23_b = map_reduce("gpt-4", b_split_2023, temperature=0)
    output_map_reduce_23_c = map_reduce("gpt-4", c_split_2023, temperature=0)
    sections_2023_summary = (
        output_map_reduce_23_a
        + "\n"
        + output_map_reduce_23_b
        + "\n"
        + output_map_reduce_23_c
    )

    if save_dir is not None:
        with open(save_dir, "w") as f:
            f.write(sections_2023_summary)
        return sections_2023_summary
    return sections_2023_summary


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
        map_reduce("gpt-4", split_2015_gpt4, save_dir=gpt4_2015_mp_fname, temperature=0)
    if not gpt4_2023_mp_fname.exists():
        map_reduce("gpt-4", split_2023_gpt4, save_dir=gpt4_2015_mp_fname, temperature=0)

    ## Davinci
    split_2015_davinci = tokenize_document(data_2015, chunk_size=1000, chunk_overlap=0)
    split_2023_davinci = tokenize_document(data_2023, chunk_size=1000, chunk_overlap=0)
    davinci_2015_mp_fname = Path(
        SUMMARY_SAVE_DIR, "davinci_map_reduce_summarized_2015.txt"
    )
    davinci_2023_mp_fname = Path(
        SUMMARY_SAVE_DIR, "davinci_map_reduce_summarized_2023.txt"
    )
    if not davinci_2015_mp_fname.exists():
        map_reduce(
            "text-davinci-003",
            split_2015_davinci,
            save_dir=davinci_2015_mp_fname,
            temperature=0,
            max_tokens=1000,
        )

    if not davinci_2023_mp_fname.exists():
        map_reduce(
            "text-davinci-003",
            split_2023_davinci,
            save_dir=davinci_2023_mp_fname,
            temperature=0,
            max_tokens=1000,
        )

    ## Sections
    ### 2015
    sections_summary_2015_fname = Path(SUMMARY_SAVE_DIR, "2015_sections_summary.txt")
    if not sections_summary_2015_fname.exists():
        map_reduce_sections_2015(data_2015, sections_summary_2015_fname)
    ## 2023
    sections_summary_2023_fname = Path(SUMMARY_SAVE_DIR, "2023_sections_summary.txt")
    if not sections_summary_2023_fname.exists():
        map_reduce_sections_2023(data_2023, sections_summary_2023_fname)


if __name__ == "__main__":
    main()
