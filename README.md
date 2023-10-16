# Terms-Conditions-Comparison
Utilizing LLMs to derive insights from changes to Terms of Conditions over time. 


# Summarization notes

Stuff didnt work well, basicaly just said "These are Apple's T&Cs" - lost almost all info. 

2023 summarisation worked well with langchains default prompt for map reduce. 
2015 didnt work as well with - 
*  CharacterTextSplitter 
*  default prompt

there seems to be a text limit, where beyond a ceetain length the summarisatio doesnt work
maybe catastrophic fogetting?

following prompt works better:

"reduce_template = """Write a concise summary of the following text delimited by triple backquotes.
Return your response in bullet points which covers the key points of the text.
```{doc_summaries}```
BULLET POINT SUMMARY:""""

swapping to recursive splitter works better

chunk size of 1000 causes return to be cut off, upping to 4000 works better

ALso trying best vector summarization as described in here

https://github.com/gkamradt/langchain-tutorials/blob/main/data_generation/5%20Levels%20Of%20Summarization%20-%20Novice%20To%20Expert.ipynb


## Davinci

davinci in summarisation doesnt work well