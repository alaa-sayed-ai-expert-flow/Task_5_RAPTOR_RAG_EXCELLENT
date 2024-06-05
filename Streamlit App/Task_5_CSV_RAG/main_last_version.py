import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm 
import pandas as pd
import json
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer , util
import tiktoken
import numpy as np
import matplotlib.pyplot as plt
from spacy.lang.en import English
from langchain.docstore.document import Document
import torch
import os
import ast
from deep_translator import GoogleTranslator
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.llms import GPT4All
from spacy.lang.en import English

from langchain.memory import ConversationBufferMemory
from time import perf_counter as timer
from langchain import PromptTemplate, LLMChain
import streamlit as st
device = "cuda" if torch.cuda.is_available() else "cpu"
import textwrap

def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container,  initial_text="" , token_no = 0):
        self.container = container
        self.text = initial_text
        self.token_no = token_no
       

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.token_no = self.token_no+1
        # Add the response to the chat window
        with self.container.empty():
            #self.container.header("Chat Session")
            with self.container.chat_message("ai"): 
                #print(str(self.token_no))
                if self.token_no != 0:
                    self.text += token
                    print(token , end = '',flush = True)
                    #print(self.text)
                    st.markdown(self.text)
                else:
                    self.text = ""

# Loading Json file
def read_json_file(file_path):
    with open(file_path, 'r') as json_file:
        json_data = json.load(json_file)
    return json_data


## Helper Fuction to count the number of Tokensin each text
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def dot_product(vector1, vector2):
    return torch.dot(vector1, vector2)

def cosine_similarity(vector1, vector2):
    dot_product = torch.dot(vector1, vector2)

    # Get Euclidean/L2 norm of each vector (removes the magnitude, keeps direction)
    norm_vector1 = torch.sqrt(torch.sum(vector1**2))
    norm_vector2 = torch.sqrt(torch.sum(vector2**2))

    return dot_product / (norm_vector1 * norm_vector2)

def retrieve_relevant_resources(query: str,
                                embeddings: torch.tensor,
                                model,
                                n_resources_to_return: int=5,
                                print_time: bool=True):
    """
    Embeds a query with model and returns top k scores and indices from embeddings.
    """

    # Embed the query
    query_embedding = model.encode(query, 
                                   convert_to_tensor=True) 

    # Get dot product scores on embeddings
    start_time = timer()
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    end_time = timer()

    if print_time:
        print(f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end_time-start_time:.5f} seconds.")

    scores, indices = torch.topk(input=dot_scores, 
                                 k=n_resources_to_return)

    return scores, indices

def print_top_results_and_scores(query: str,
                                 embeddings: torch.tensor,
                                 pages_and_chunks,
                                 n_resources_to_return: int=5):
    """
    Takes a query, retrieves most relevant resources and prints them out in descending order.

    Note: Requires pages_and_chunks to be formatted in a specific way (see above for reference).
    """
    
    scores, indices = retrieve_relevant_resources(query=query,
                                                  model = st.session_state["embedding_model"],
                                                  embeddings=embeddings,
                                                  n_resources_to_return=n_resources_to_return)
    
    print(f"Query: {query}\n")
    print("Results:")
    # Loop through zipped together scores and indicies
    for score, index in zip(scores, indices):
        print(f"Score: {score:.4f}")
        # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
        print_wrapped(pages_and_chunks[index]["sentence_chunks"])
        # Print the page number too so we can reference the textbook further and check the results
        print(f"Page number: {pages_and_chunks[index]['source']}")
        print(f"Link: {pages_and_chunks[index]['link']}")
        print(f"Title: {pages_and_chunks[index]['title']}")
        print("\n")
def create_chain():
    meta_list = []
    # Create Simple List for chat history
    chat_history = []
    final_file_path = 'sentence_chunks_emb.csv'
    if os.path.isfile(final_file_path):
        # Import texts and embedding df
        text_chunks_and_embedding_df = pd.read_csv("sentence_chunks_emb.csv")
        # Convert the string representation to actual lists
        text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(ast.literal_eval)
        embedding_array = np.array(text_chunks_and_embedding_df["embedding"])
        # Define the maximum sequence length
        max_seq_length = max(len(seq) for seq in embedding_array)
        print(max_seq_length)
        # Pad or truncate sequences to the maximum length
        processed_arr = np.array([seq + [0.0] * (max_seq_length - len(seq)) for seq in embedding_array])
        # Check if the array is of float type
        if processed_arr.dtype == np.float32 or processed_arr.dtype == np.float64:
            print("The array is of float type")
        else:
            print("The array is not of float type")

        embeddings = torch.tensor(processed_arr, dtype=torch.float32).to(device)
        # Convert texts and embedding df to list of dicts
        pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")
        model_name = "intfloat/multilingual-e5-large" # Max Seq Len = 512
        embedding_model = SentenceTransformer(model_name_or_path=model_name, device="cuda") # choose the device to load the model to (note: GPU will often be *much* faster than CPU)
        stream_handler = StreamHandler(st.sidebar.empty())
        callback_manager = CallbackManager([stream_handler])
        # Loading Model
        llm = GPT4All(
            model="C:/Users/thegh/Python Projects/Ai Models/alaa_ai_model_mistral_v1.9.gguf",
            # verbose=True,
            #callback_manager = callback_manager
            streaming = True,
            device = 'gpu',callback_manager = callback_manager , max_tokens = 6000 , temp = 0.1 , n_batch = 64
        )
        

        return llm , chat_history , embedding_model , embeddings , pages_and_chunks
    else:
        filePath = 'text_chunks_and_embeddings_df.csv'
        if os.path.isfile(filePath):
            # Import saved file and view
            text_chunks_and_embedding_df = pd.read_csv(filePath)
            text_chunks_and_embedding_df["embedding"].replace(np.nan , "[[0]]" , inplace = True)
            text_chunks_and_embedding_df["sentence_chunks"].replace(np.nan , "[]" , inplace = True)
            # Convert the string representation to actual lists
            text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(ast.literal_eval)
            # Convert the column with lists to a single list
            single_list = text_chunks_and_embedding_df["embedding"].explode()
            single_list = text_chunks_and_embedding_df["embedding"].explode().reset_index(drop=True)
            text_chunks_and_embedding_df["sentence_chunks"] = text_chunks_and_embedding_df["sentence_chunks"].apply(ast.literal_eval)
            sentence_chunk_single_list = text_chunks_and_embedding_df["sentence_chunks"].explode()
            sentence_chunk_single_list = text_chunks_and_embedding_df["sentence_chunks"].explode().reset_index(drop=True)
            metadata_df = text_chunks_and_embedding_df["metadata"]
            text_chunks_and_embedding_df["metadata"] = text_chunks_and_embedding_df["metadata"].apply(ast.literal_eval)
            sentence_chunk_single_list_meta = text_chunks_and_embedding_df["sentence_chunks"].explode()
            for chunk in sentence_chunk_single_list_meta.index:
                meta_list.append(text_chunks_and_embedding_df["metadata"][chunk])
                print(text_chunks_and_embedding_df["metadata"][chunk])
                print('\n')
            meta_data_df = pd.DataFrame(meta_list)
            sentence_chunks_emb_df = pd.DataFrame({'sentence_chunks': sentence_chunk_single_list , 'embedding': single_list})
            # Combine the original DataFrame with the normalized metadata DataFrame
            combined_df = pd.concat([sentence_chunks_emb_df, meta_data_df], axis=1)
            combined_df.to_csv("sentence_chunks_emb.csv" , index=False)
            create_chain()
        else:
            nltk.download('punkt')  # Download necessary resource for NLTK
            json_data = read_json_file("general_file_posts_pages.json")
            # Create DataFrame
            df = pd.DataFrame(json_data)

            print("Note: metadata contains (source , title , link , page_number , hour_of_update , minute_of_update , post_number) ... Nan is between Page_number and Post_number when you find Nan in our Dataframe that means that one of them has a value not both ...")

            # Normalize the 'metadata' column
            metadata_df = pd.json_normalize(df['metadata'])

            # Combine the original DataFrame with the normalized metadata DataFrame
            combined_df = pd.concat([df.drop(columns=['metadata']), metadata_df], axis=1)
            nlp = English()

            # Add a sentencizer pipeline, see https://spacy.io/api/sentencizer/ 
            nlp.add_pipe("sentencizer")
            
            for item in tqdm(json_data):
                item["sentences"] = list(nlp(item["page_content"]).sents)
                
                item["sentences"] = [str(sentence) for sentence in item["sentences"]]
                
                # Count the sentences 
                item["page_sentence_count_spacy"] = len(item["sentences"])

                item["document_character_length"] = len(item["page_content"])
                item["document_tokens_nltk"] = len(word_tokenize(item["page_content"]))
                item["document_tokens_tiktoken"] = num_tokens_from_string(item["page_content"], "cl100k_base")
                print("\n")
                print("We 've here in That Document : "+str(len(item['sentences']))+" Sentences ")
                # single_chunk_tokens = [len(word_tokenize(item)) for item in tqdm(chunks)]
                # print(single_chunk_tokens)
                # print("The Max number of tokens from the biggest chunk : " + str(max(single_chunk_tokens)) + " Tokens ")
                print("==========================================================================================")
                print("\n")
            content_chars = [len(item["page_content"]) for item in tqdm(json_data)]
            print("The Max Characters Document in Whole Documents : " + str(max(content_chars)))
            content_tokens = [len(word_tokenize(item["page_content"])) for item in tqdm(json_data)]
            print("The Max Tokens Document in Whole Documents : " + str(max(content_tokens)))
            counts = [num_tokens_from_string(item["page_content"], "cl100k_base") for item in tqdm(json_data)]
            # Loading Embedding Model
            model_name = "intfloat/multilingual-e5-large" # Max Seq Len = 512
            embedding_model = SentenceTransformer(model_name_or_path=model_name, device="cuda")
            print(f"Model 's Maximum Sequence Length : {SentenceTransformer(model_name).max_seq_length}")
            single_list_text_chunks = []
            documents_and_chunks = json_data
            r_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1000,
                chunk_overlap = 250,
                #separators = [" ."]#["\n"]# , "\n" , " " , ""]
            )
            
            for item in tqdm(documents_and_chunks):
                chunks = r_splitter.split_text(str(item['page_content']))
                if len(chunks) > 0:
                    item["sentence_chunks"] = chunks
                    #print(item["sentence_chunks"])
                    #print("\n")
                    item["num_chunks"] = len(chunks)
                    #print("We 've here in That Document : "+str(len(item["sentence_chunks"]))+" Chunks ")
                    single_chunk_tokens = [len(word_tokenize(item)) for item in tqdm(chunks)]
                    item["single_chunk_tokens"] = single_chunk_tokens
                    item["max_single_chunk_token"] = max(single_chunk_tokens)
                    #print(single_chunk_tokens)
                    #print("The Max number of tokens from the biggest chunk : " + str(max(single_chunk_tokens)) + " Tokens ")
                    #print("==========================================================================================")
                    #print("\n")
                else:
                    print("\nThat Document has no data : \n")
                    print(item)
                    print("\n")
                    print("==========================================================================================")
            

            # Turn text chunks into a single list
            single_list_text_chunks = [chunk for item in (r_splitter.split_text(item['page_content']) for item in tqdm(documents_and_chunks)) for chunk in item]
            print("We've "+str(len(single_list_text_chunks)) + " Chunks ")
            content_tokens = [len(word_tokenize(item)) for item in tqdm(single_list_text_chunks)]
            print("The Max Tokens in Whole Chunks : " + str(max(content_tokens)))
            
            # Sentences are encoded/embedded by calling model.encode()
            embedding_model.to("cuda")
            # Create embeddings one by one on the GPU
            for item in tqdm(documents_and_chunks):
                if "sentence_chunks" in item:
                    item["embedding"] = embedding_model.encode(item["sentence_chunks"]).tolist()
                else:
                    print("\n")
                    print(item)
                    print("That Document " + item["page_content"] + " has no chunks to be embedded")
            # Save embeddings to file
            text_chunks_and_embeddings_df = pd.DataFrame(documents_and_chunks)
            embeddings_df_save_path = "text_chunks_and_embeddings_df.csv"
            text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)
            create_chain()







# Set the webpage title
st.set_page_config(
    page_title="Pages & Posts s' Chat Robot!"
)

# Create a header element
st.header("Pages & Posts s'Chat Robot!")



# Create Select Box for target Language
lang_opts = ["ar","en" , "fr" , "pa"]
lang_selected = st.selectbox("Select Target Language " , options = lang_opts)





# We store the conversation in the session state.
# This will be used to render the chat conversation.
# We initialize it with the first message we want to be greeted with.
if "messages" not in st.session_state:
    # Create LLM chain to use for our chatbot.
    st.session_state["llm"] , st.session_state["chat_history"] , st.session_state["embedding_model"] , st.session_state["embeddings"] , st.session_state["pages_and_chunks"]= create_chain()
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I help you today?"}
    ]

if "current_response" not in st.session_state:
    st.session_state.current_response = ""

# We loop through each message in the session state and render it as
# a chat message.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])




# We take questions/instructions from the chat input to pass to the LLM
if user_prompt := st.chat_input("Your message here", key="user_input"):

    # Add our input to the session state
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )

    # Add our input to the chat window
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Pass our input to the LLM chain and capture the final responses.
    # It is worth noting that the Stream Handler is already receiving the
    # streaming response as the llm is generating. We get our response
    # here once the LLM has finished generating the complete response.
    user_prompt = GoogleTranslator(source='auto', target='en').translate(user_prompt)
    query_embedding = st.session_state["embedding_model"].encode(user_prompt, convert_to_tensor=True)
    start_time = timer()
    dot_scores = util.dot_score(a=query_embedding, b=st.session_state["embeddings"])[0]
    end_time = timer()
    print(f"Time take to get scores on {len(st.session_state['embeddings'])} embeddings: {end_time-start_time:.5f} seconds.")

    # 4. Get the top-k results (we'll keep this to 5)
    top_results_dot_product = torch.topk(dot_scores, k=5)
    larger_embeddings = torch.randn(100*st.session_state["embeddings"].shape[0], 1024).to(device)
    print(f"Embeddings shape: {larger_embeddings.shape}")

    # Perform dot product across 168,000 embeddings
    start_time = timer()
    dot_scores = util.dot_score(a=query_embedding, b=larger_embeddings)[0]
    end_time = timer()
    print(f"Time take to get scores on {len(larger_embeddings)} embeddings: {end_time-start_time:.5f} seconds.")
    
    print("Results:")
    # Loop through zipped together scores and indicies from torch.topk
    for score, idx in zip(top_results_dot_product[0], top_results_dot_product[1]):
        print(f"Score: {score:.4f}")
        # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
        print("Text:")
        print_wrapped(st.session_state["pages_and_chunks"][idx]["sentence_chunks"])
        # Print the page number too so we can reference the textbook further (and check the results)
        print(f'Page number: {st.session_state["pages_and_chunks"][idx]["source"]}')
        print(f'Link: {st.session_state["pages_and_chunks"][idx]["link"]}')
        print(f'Title: {st.session_state["pages_and_chunks"][idx]["title"]}')
        print("\n")
    # Get just the scores and indices of top related results
    scores, indices = retrieve_relevant_resources(query=user_prompt,
                                                  model = st.session_state["embedding_model"],
                                              embeddings=st.session_state["embeddings"])
    # Print out the texts of the top scores
    print_top_results_and_scores(query=user_prompt,
                                 pages_and_chunks = st.session_state["pages_and_chunks"],
                                 embeddings=st.session_state["embeddings"])
    # Create a list of context items
    context_items = [st.session_state["pages_and_chunks"][i] for i in indices]
    # Join context items into one dotted paragraph
    context = "- " + "\n- ".join([item["sentence_chunks"] for item in context_items])
##    template = """<s>[INST] You are a helpful, respectful and honest assistant. Answer in details from the context
##        Answer the question below from context below :
##        {context}
##        {question} [/INST] </s>
##        """
    template = """
    <s>[INST] You are a helpful assistant, you will use the provided context to answer user questions.
    Read the given context before answering questions and think step by step. If you can not answer a user question based on the provided context,
    inform the user. Do not use any other information for answering user. Provide a detailed answer to the question.
    Context : {context}
    Question : {question} [/INST] </s>
    """
    prompt = PromptTemplate(template=template, input_variables=["question","context"])
    llm_chain = LLMChain(prompt=prompt, llm=st.session_state["llm"])
    
    response = llm_chain.run({"question":user_prompt,"context":context})
    
##    print("Current buffer length " + str(len(st.session_state["chat_history"])))
##
##    
##    print("Before pop : " + str(st.session_state["chat_history"]))
##    if len(st.session_state["chat_history"]) > 1:
##        del st.session_state["chat_history"][:1]
##        #buffer.pop(0)
##
##    print("After pop : " + str(st.session_state["chat_history"]))
##    
##    st.session_state["chat_history"].append((user_prompt , response))
    
    
    
    response = GoogleTranslator(source='auto', target=lang_selected).translate(response)
    
    # Add the response to the session state
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    # Add the response to the chat window
    with st.chat_message("assistant"):
        st.markdown(response)





