import warnings
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient

openai_api_key = '<Api Key>'
mongodb_conn_string = 'mongodb+srv://<username>:<password>@maincluster.d67gxdl.mongodb.net/'
db_name = "<db_name>"
collection_name = "<collection_name>"
index_name = "<vsearch_index_name>"

# Filter out the UserWarning from langchain
warnings.filterwarnings("ignore", category=UserWarning, module="langchain.chains.llm")

# Step 1: Load
loaders = [
    WebBaseLoader("https://threadwaiting.com/python-oops/"),
    WebBaseLoader("https://threadwaiting.com/semantic-search-with-facebook-ai-similarity-search-faiss/")

]
data = []
for loader in loaders:
    data.extend(loader.load())

# Step 2: Transform (Split)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=[
                                               "\n\n", "\n", "(?<=\. )", " "], length_function=len)
docs = text_splitter.split_documents(data)
print('Split into ' + str(len(docs)) + ' docs')

# Step 3: Embed
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Step 4: Store
# Initialize MongoDB python client
client = MongoClient(mongodb_conn_string)
collection = client[db_name][collection_name]

# Reset w/out deleting the Search Index 
collection.delete_many({})

# Insert the documents in MongoDB Atlas with their embedding
docsearch = MongoDBAtlasVectorSearch.from_documents(
    docs, embeddings, collection=collection, index_name=index_name
)

# Process arguments

questions = ["What is an object?", "What is FAISS?"]

# initialize vector store
vectorStore = MongoDBAtlasVectorSearch(
    collection, OpenAIEmbeddings(openai_api_key=openai_api_key), index_name=index_name
)

# perform a similarity search between the embedding of the query and the embeddings of the documents
# print("\nQuery Response:")
for query in questions:
    print("---------------")
    print(query)
    docs = vectorStore.max_marginal_relevance_search(query, K=1)
    if len(docs)>0:
        print(docs[0].metadata['title'])
        print(docs[0].page_content)

for query in questions:
    # Contextual Compression
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorStore.as_retriever()
    )

    print("\nAI Response:")
    print("-----------")
    print(query)
    compressed_docs = compression_retriever.get_relevant_documents(query)
    if len(compressed_docs)>0:
        print(compressed_docs[0].metadata['title'])
        print(compressed_docs[0].page_content)