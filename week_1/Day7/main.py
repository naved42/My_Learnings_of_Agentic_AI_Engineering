from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ============================================================
# 1. DATABASE
# ============================================================

# Example: SQLite
db = SQLDatabase.from_uri("sqlite:///my_database.db")


# ============================================================
# 2. LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# ============================================================
# 3. STATE
# ============================================================

class AgentState(TypedDict):
    question: str
    schema: str
    sql: str
    result: str
    answer: str
    error: str


# ============================================================
# 4. GET DATABASE SCHEMA
# ============================================================

def get_schema(state: AgentState):

    schema = db.get_table_info()

    return {
        "schema": schema,
        "error": ""
    }


# ============================================================
# 5. GENERATE SQL
# ============================================================

def generate_sql(state: AgentState):

    question = state["question"]
    schema = state["schema"]

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an expert SQL developer.

Your job is to generate a SQL query that answers
the user's question.

Use ONLY the tables and columns provided in the schema.

Rules:
- Generate valid SQL.
- Do not invent tables.
- Do not invent columns.
- Do not use markdown.
- Return ONLY the SQL query.
- Do not explain the query.

Database schema:

{schema}
"""
        ),
        (
            "human",
            """
User question:

{question}
"""
        )
    ])

    chain = prompt | llm

    response = chain.invoke({
        "schema": schema,
        "question": question
    })

    sql = response.content.strip()

    # Remove markdown fences if the LLM returns them
    sql = sql.replace("```sql", "")
    sql = sql.replace("```", "")
    sql = sql.strip()

    return {
        "sql": sql,
        "error": ""
    }


# ============================================================
# 6. VALIDATE SQL
# ============================================================

def validate_sql(state: AgentState):

    question = state["question"]
    schema = state["schema"]
    sql = state["sql"]

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an expert SQL reviewer.

Review the SQL query below.

Check:
1. Does it answer the user's question?
2. Are all tables valid?
3. Are all columns valid?
4. Is the SQL syntax valid?
5. Does it use only the provided schema?

If the SQL is correct:
return the SQL unchanged.

If it is incorrect:
return a corrected SQL query.

IMPORTANT:
Return ONLY SQL.
Do not use markdown.
Do not explain anything.

Schema:

{schema}
"""
        ),
        (
            "human",
            """
User question:
{question}

SQL:
{sql}
"""
        )
    ])

    chain = prompt | llm

    response = chain.invoke({
        "schema": schema,
        "question": question,
        "sql": sql
    })

    validated_sql = response.content.strip()

    validated_sql = validated_sql.replace("```sql", "")
    validated_sql = validated_sql.replace("```", "")
    validated_sql = validated_sql.strip()

    return {
        "sql": validated_sql,
        "error": ""
    }


# ============================================================
# 7. EXECUTE SQL
# ============================================================

def execute_sql(state: AgentState):

    sql = state["sql"]

    try:
        result = db.run(sql)

        print("\n--- SQL EXECUTION ---")
        print("SQL:", sql)
        print("RESULT:", result)

        return {
            "result": str(result),
            "error": ""
        }

    except Exception as e:
        print("\n--- SQL ERROR ---")
        print(e)

        return {
            "result": "",
            "error": str(e)
        }


# ============================================================
# 8. ANALYZE RESULT
# ============================================================

def analyze_result(state: AgentState):

    question = state["question"]
    result = state["result"]

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a data analyst.

Analyze the SQL result and determine the answer
to the user's question.

Be accurate.
Do not invent information.

Return a concise explanation of what the result means.
"""
        ),
        (
            "human",
            """
Question:
{question}

SQL result:
{result}
"""
        )
    ])

    chain = prompt | llm

    response = chain.invoke({
        "question": question,
        "result": result
    })

    return {
        "answer": response.content.strip()
    }


# ============================================================
# 9. GENERATE FINAL ANSWER
# ============================================================

def generate_answer(state: AgentState):

    question = state["question"]
    result = state["result"]
    analysis = state["answer"]

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a helpful database assistant.

Answer the user's question using the SQL result
and analysis.

Rules:
- Be concise.
- Do not mention internal agent steps.
- Do not mention prompts.
- Do not invent data.
- If appropriate, include the relevant numbers.
"""
        ),
        (
            "human",
            """
Question:
{question}

SQL result:
{result}

Analysis:
{analysis}
"""
        )
    ])

    chain = prompt | llm

    response = chain.invoke({
        "question": question,
        "result": result,
        "analysis": analysis
    })

    return {
        "answer": response.content.strip()
    }


# ============================================================
# 10. BUILD GRAPH
# ============================================================

graph = StateGraph(AgentState)


graph.add_node("schema", get_schema)

graph.add_node("generate_sql", generate_sql)

graph.add_node("validate", validate_sql)

graph.add_node("execute", execute_sql)

graph.add_node("analyze", analyze_result)

graph.add_node("answer", generate_answer)


# ============================================================
# 11. EDGES
# ============================================================

graph.add_edge(START, "schema")

graph.add_edge("schema", "generate_sql")

graph.add_edge("generate_sql", "validate")

graph.add_edge("validate", "execute")

graph.add_edge("execute", "analyze")

graph.add_edge("analyze", "answer")

graph.add_edge("answer", END)


# ============================================================
# 12. COMPILE
# ============================================================

app = graph.compile()


# ============================================================
# 13. RUN AGENT
# ============================================================

question = "How many customers are in the database?"

result = app.invoke({
    "question": question,
    "schema": "",
    "sql": "",
    "result": "",
    "answer": "",
    "error": ""
})


print("\nFINAL ANSWER:")
print(result["answer"])

print("\nGENERATED SQL:")
print(result["sql"])

print("\nSQL RESULT:")
print(result["result"])
