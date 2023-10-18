from pathlib import Path

from dotenv import load_dotenv
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate

from create_summaries import instantiate_model, load_template

SAVE_DIR = Path("data", "standardisation")
SUMMARIES_DIR = Path("data", "summaries")


def create_prompt():
    query = load_template(Path("prompts_templates", "standardisation_prompt.txt"))
    # prompt_template = PromptTemplate(input_variables=["text1", "text2"], template=query)
    human_message_prompt = HumanMessagePromptTemplate.from_template(query)
    chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
    return chat_prompt

def main():
    load_dotenv()
    sections_2015_fname = Path(SUMMARIES_DIR, "2015_sections_summary.txt")
    sections_2023_fname = Path(SUMMARIES_DIR, "2023_sections_summary.txt")
    with open(sections_2015_fname, "r", encoding="utf-8") as f:
        section_2015 = f.read()
    with open(sections_2023_fname, "r", encoding="utf-8") as f:
        section_2023 = f.read()

    chat_prompt = create_prompt()

    gpt35_standardisation_fname = Path(SAVE_DIR, "gpt35_standardisation_advice.txt")
    if not gpt35_standardisation_fname.exists():
        gpt35 = instantiate_model("gpt-3.5-turbo-0613")
        gpt35_output = gpt35(
            chat_prompt.format_prompt(text1=section_2015, text2=section_2023).to_messages()
        )
        with open(gpt35_standardisation_fname, "w", encoding="utf-8") as f:
            f.write(gpt35_output.content)


    gpt4_standardisation_fname = Path(SAVE_DIR, "gpt4_standardisation_advice.txt")
    if not gpt4_standardisation_fname.exists():
        gpt4 = instantiate_model("gpt-4")
        gpt4_output = gpt4(
            chat_prompt.format_prompt(text1=section_2015, text2=section_2023).to_messages()
        )
        with open(gpt4_standardisation_fname, "w", encoding="utf-8") as f:
            f.write(gpt4_output.content)


if __name__ == "__main__":
    main()
