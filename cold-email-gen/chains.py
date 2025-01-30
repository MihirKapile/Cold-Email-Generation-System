import os

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()


class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-70b-versatile",
        )

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, resume):
        prompt_email = PromptTemplate.from_template(
            """
          ### JOB DESCRIPTION:
          {job_description}

          ### INSTRUCTION:
          You are Mihir Milind Kapile, a Software Engineer. You have a Master’s in Computer Science and extensive experience in software development, including backend development with Django, frontend work with React.js, and DevOps with Azure. 
          You have successfully completed projects like the LLM Assisted Mobile App Testing Platform, a sophisticated web application for research paper and patent sales, and a Face Emotion Detection algorithm. 
          Your job is to write a cold email to the client regarding the job mentioned above, describing your capability in fulfilling their needs. 
          Also, highlight relevant projects from your resume {resume} to showcase your expertise.

          Remember, you are Mihir Milind Kapile, a Software Engineer.
          Do not provide a preamble.
          ### EMAIL (NO PREAMBLE):
        """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "resume": resume})
        return res.content
