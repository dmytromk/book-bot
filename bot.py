import os
import ast
import re
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import SQLDatabase


class LangChainAgent:
    def __init__(self, db_uri, api_key):
        os.environ["OPENAI_API_KEY"] = api_key
        self.db = SQLDatabase.from_uri(db_uri)
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.tools = self.toolkit.get_tools()
        self.system_message = SystemMessage(content=self._create_system_message())

        self.books = self._query_as_list('SELECT name FROM books')
        self.authors = self._query_as_list('SELECT name FROM authors')
        self.publishers = self._query_as_list('SELECT name FROM publishers')

        self.vector_db = FAISS.from_texts(self.authors + self.books + self.publishers, OpenAIEmbeddings())
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 5})
        self._add_retriever_tool()

        self.agent = create_react_agent(self.llm, self.tools, messages_modifier=self.system_message)

    def _create_system_message(self):
        table_names = self.db.get_usable_table_names()
        return f"""You are an agent designed to interact with PostgreSQL database.

        Given an input question, create a syntactically correct PostgreSQL query to run, then look at the results 
        of the query and return the answer. Unless the user specifies a specific number of examples they wish to obtain, 
        always limit your query to at most 5 results. You can order the results by a relevant column to return the most 
        interesting examples in the database. Never query for all the columns from a specific table, only ask for the relevant 
        columns given the question. You have access to tools for interacting with the database.

        Only use the given tools. Only use the information returned by the tools to construct your final answer.
        You MUST double check your query before executing it. If you get an error while executing a query, rewrite the 
        query and try again.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

        You have access to the following tables: {table_names}

        If you need to filter on a proper noun, you must ALWAYS first look up the filter value using the "search_proper_nouns" 
        tool! Do not try to guess at the proper name - use this function to find similar ones."""

    def _query_as_list(self, query):
        res = self.db.run(query)
        if res:
            res = [el for sub in ast.literal_eval(res) for el in sub if el]
            res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
        return list(set(res))

    def _add_retriever_tool(self):
        description = """Use to look up values to filter on. Input is an approximate spelling of the proper noun, output is 
        valid proper nouns. Use the noun most similar to the search."""

        retriever_tool = create_retriever_tool(
            self.retriever,
            name="search_proper_nouns",
            description=description,
        )
        self.tools.append(retriever_tool)

    def query(self, question):
        answer = ""
        for response in self.agent.stream({"messages": [HumanMessage(content=question)]}):
            print(answer)
            answer = response
        return answer

