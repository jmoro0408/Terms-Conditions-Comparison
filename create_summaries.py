from pathlib import Path
from typing import Optional, Union

import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.document_loaders import Docx2txtLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

SUMMARY_SAVE_DIR = Path("data", "summaries")
DATA_DIR = Path("data", "raw")


def load_document(document_fname: Union[Path, str]) -> list[Document]:
    """Loads a .docx document into a Langchain Document

    Args:
        document_fname (Union[Path, str]): filepath of .docx

    Returns:
        list[Document]: Langchain Document
    """
    return Docx2txtLoader(str(document_fname)).load()  # str reqd for loader


def load_template(template_fname: Union[Path, str]) -> str:
    """Loads a .txt file into a str

    Args:
        template_fname (Union[Path, str]): filepath of .txt file

    Returns:
        str: .tt file content
    """
    with open(template_fname, "r", encoding="utf-8") as f:
        return f.read()


def instantiate_model(model_name: str, **kwargs) -> Union[ChatOpenAI, OpenAI]:
    """Create a ChatOpenAI or OpenAI object from a model_name

    Args:
        model_name (str): Model to use. Should be one of "gpt-4",
        "text-davinci-003", or "gpt-3.5-turbo-0613"

    Returns:
        Union[ChatOpenAI, OpenAI]: Either ChatOpenAi or OpenAI object depending on the model name provided.
    """
    if model_name == "gpt-4":
        return ChatOpenAI(model_name="gpt-4", temperature=0, **kwargs)
    if model_name == "text-davinci-003":
        return OpenAI(model="text-davinci-003", temperature=0, **kwargs)
    if model_name == "gpt-3.5-turbo-0613":
        return ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0, **kwargs)
    raise NotImplementedError("Supplied model name is not yet implemented.")


def map_reduce(
    model_name: str,
    text_split: list[Document],
    save_dir: Optional[Union[Path, str]] = None,
    **kwargs
) -> str:
    """Map reduce summarisation as implemented here:
    https://python.langchain.com/docs/modules/chains/document/map_reduce

    Note map_reduce prompt filenames are hardcoded, a "map_template.txt" and
    "reduce_template.txt" file is expected in the "prompts_templates" folder.

    Args:
        model_name (str): name of model to use, see instantiate_model func for further info
        text_split (list[Document]): Input text as langchain Document
        save_dir (Optional[Union[Path, str]], optional):
            directory to save summarisation. Defaults to None.

    Returns:
        str: summary
    """
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
        with open(save_dir, "w", encoding="utf-8") as f:
            f.write(output)
        return output
    return output


def tokenize_document(
    input_document: list[Document], chunk_size: int, chunk_overlap: int
):
    """
    Tokenizes a document into smaller chunks for processing.

    Args:
        input_document (list[Document]): List of documents to be tokenized.
        chunk_size (int): Size of each tokenized chunk.
        chunk_overlap (int): Overlap between adjacent chunks.

    Returns:
        list[Document]: List of tokenized document chunks.

    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(input_document)


def string_between_substrings(input_str: str, start: str, end: Optional[str]):
    """
    Extracts the substring between 'start' and 'end' within 'input_str'.
    If 'end' is not provided, it returns the substring starting from 'start'
    to the end of 'input_str'.

    Args:
        input_str (str): The input string to search within.
        start (str): The starting substring.
        end (Optional[str]): The ending substring (optional).

    Returns:
        str: The substring between 'start' and 'end' in 'input_str'.
    """
    if end is None:
        return (input_str.split(start))[1]
    return (input_str.split(start))[1].split(end)[0]


def map_reduce_sections_2015(
    document_2015: list[Document], save_dir: Optional[Union[Path, str]] = None
) -> str:
    """
    Splits the 2015 T&Cs into sections and performs map-reduce summarisation

    Args:
        document_2015 (list[Document]): 2015 T&Cs as a langchain Document
        save_dir (Optional[Union[Path, str]], optional): directiory to save summary in.
            Defaults to None.

    Returns:
        str: summary text
    """
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

    output_map_reduce_15_a = map_reduce("gpt-4", a_split_2015)

    output_map_reduce_15_b = map_reduce("gpt-4", b_split_2015)

    output_map_reduce_15_c = map_reduce("gpt-4", c_split_2015)
    sections_2015_summary = (
        output_map_reduce_15_a
        + "\n"
        + output_map_reduce_15_b
        + "\n"
        + output_map_reduce_15_c
    )
    if save_dir is not None:
        with open(save_dir, "w", encoding="utf-8") as f:
            f.write(sections_2015_summary)
        return sections_2015_summary
    return sections_2015_summary


def map_reduce_sections_2023(
    document_2023: list[Document], save_dir: Optional[Union[Path, str]] = None
):
    """
    Splits the 2023 T&Cs into sections and performs map-reduce summarisation

    Args:
        document_2023 (list[Document]): 2023 T&Cs as a langchain Document
        save_dir (Optional[Union[Path, str]], optional): directiory to save summary in. Defaults to None.

    Returns:
        str: summary text
    """
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
        with open(save_dir, "w", encoding="utf-8") as f:
            f.write(sections_2023_summary)
        return sections_2023_summary
    return sections_2023_summary


def summarize_with_vectors(
    model_name: str,
    document_split: list[Document],
    num_clusters: int,
    save_dir: Optional[Union[Path, str]] = None,
):
    """
    Interesting approach to summarizing large documents, inspired from this blog post:
    https://pashpashpash.substack.com/p/tackling-the-challenge-of-document

    Args:
        model_name (str): The name of the language model to use, shoudl be either
        "gpt-3.5-turbo-0613" or 'gpt-4'
        document_split (list[Document]): Langchain document to summarize.
        num_clusters (int): Number of clusters to create for document grouping.
        save_dir (Optional[Union[Path, str]]): Optional directory to save the summary.

    Returns:
        str: The summarized document.

    """

    llm = ChatOpenAI(model_name=model_name)

    map_template_fname = Path("prompts_templates", "map_template.txt")
    map_template = load_template(map_template_fname)
    map_prompt = PromptTemplate.from_template(map_template)

    reduce_template_fname = Path("prompts_templates", "reduce_template.txt")
    reduce_template = load_template(reduce_template_fname)
    reduce_prompt = PromptTemplate.from_template(reduce_template)

    embeddings = OpenAIEmbeddings()
    vectors = np.array(
        embeddings.embed_documents([x.page_content for x in document_split])
    )
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto").fit(
        vectors
    )
    # Find the closest embeddings to the centroids

    closest_indices = []

    for i in range(num_clusters):
        # Get the list of distances from that particular cluster center
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)

        # Find the list position of the closest one (using argmin to find the smallest distance)
        closest_index = np.argmin(distances)

        # Append that position to your closest indices list
        closest_indices.append(closest_index)
    selected_indices = sorted(closest_indices)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)
    selected_docs = [document_split[doc] for doc in selected_indices]
    # Make an empty list to hold your summaries
    summary_list = []

    # Loop through a range of the lenght of your selected docs
    for i, doc in enumerate(selected_docs):
        # Go get a summary of the chunk
        chunk_summary = map_chain.run([doc])

        # Append that summary to your list
        summary_list.append(chunk_summary)

    summaries = "\n".join(summary_list)

    # Convert it back to a document
    summaries = Document(page_content=summaries)
    output = reduce_chain.run([summaries])
    if save_dir is not None:
        with open(save_dir, "w", encoding="utf-8") as f:
            f.write(output)
        return output
    return output


def plot_tsne(
    document_split: list[Document],
    num_clusters: int,
    perplexity: int,
    plot_kwargs: dict,
    save_dir: Optional[Union[Path, str]] = None,
):
    colours = [
        "maroon",
        "purple",
        "green",
        "blue",
        "yellow",
        "black",
        "aqua",
        "coral",
        "darkblue",
        "darkgreen",
        "darkmagenta",
        "darkslateblue",
        "deeppink",
        "dimgrey",
        "indianred",
        "mediumslateblue",
        "olivedrab",
        "orangered",
        "palevioletred",
    ]
    colour_map = dict(enumerate(colours))

    embeddings = OpenAIEmbeddings()
    vectors = np.array(
        embeddings.embed_documents([x.page_content for x in document_split])
    )
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced_data_tsne = tsne.fit_transform(vectors)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto").fit(
        vectors
    )
    kmeans_colours = []
    for label in kmeans.labels_:
        kmeans_colours.append(colour_map[label])
    fig = go.Figure(
        data=go.Scatter(
            x=reduced_data_tsne[:, 0],
            y=reduced_data_tsne[:, 1],
            marker_color=kmeans_colours,
            hovertext=kmeans.labels_,
            mode="markers",
            name="TSNE Cluster",
        )
    )
    fig.update_layout(
        template="simple_white",
        title=plot_kwargs["title"],
        xaxis_title="TSNE 1",
        yaxis_title="TSNE 2",
    )
    if save_dir is not None:
        save_dir = Path(save_dir)
        if save_dir.suffix != "":
            save_dir = Path(
                save_dir.parent, save_dir.stem
            )  # Removing suffix if existing
        save_html = str(save_dir) + ".html"
        save_png = str(save_dir) + ".png"
        with open(save_html, "w", encoding="utf-8") as f:
            f.write(fig.to_html())
        fig.write_image(save_png)
    fig.show()


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

    # Vectors and TSNE
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=750, chunk_overlap=50
    )
    ## 2015 Summary
    split_2015 = text_splitter.split_documents(data_2015)
    tsne_2015_fname = Path("figures", "tsne_2015_fig.html")
    plot_kwargs_2015 = {"title": "2015 Contract TSNE Plot"}
    plot_tsne(
        split_2015,
        num_clusters=10,
        perplexity=20,
        plot_kwargs=plot_kwargs_2015,
        save_dir=tsne_2015_fname,
    )

    vector_2015_summary_fname = Path("data", "summaries", "vector_2015_summary.txt")
    if not vector_2015_summary_fname.exists():
        summarize_with_vectors(
            model_name="gpt-3.5-turbo",
            document_split=split_2015,
            num_clusters=5,
            save_dir=vector_2015_summary_fname,
        )

    ## 2023 Summary
    split_2023 = text_splitter.split_documents(data_2023)
    tsne_2023_fname = Path("figures", "tsne_2023_fig.html")
    plot_kwargs_2023 = {"title": "2023 Contract TSNE Plot"}
    plot_tsne(
        split_2023,
        num_clusters=10,
        perplexity=15,
        plot_kwargs=plot_kwargs_2023,
        save_dir=tsne_2023_fname,
    )

    vector_2023_summary_fname = Path("data", "summaries", "vector_2023_summary.txt")
    if not vector_2023_summary_fname.exists():
        summarize_with_vectors(
            model_name="gpt-3.5-turbo",
            document_split=split_2023,
            num_clusters=5,
            save_dir=vector_2023_summary_fname,
        )


if __name__ == "__main__":
    main()
