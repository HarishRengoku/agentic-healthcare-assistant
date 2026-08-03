"""
Online Search Tool for BAYMAX
Searches for disease prevalence, pathogens, and epidemiological data
"""

import requests
from typing import List, Dict
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


class OnlineSearchTool:
    """Search the web for epidemiological and medical data"""

    def __init__(self):
        self.model = OllamaLLM(model="llama3.2", temperature=0.3)
        # Using a simple approach with requests - no external API needed
        self.search_queries = []

    def search_disease_in_region(self, disease_name: str, region: str) -> str:
        """
        Search for disease prevalence and info in a specific region

        Args:
            disease_name: Name of the disease (e.g., "dengue fever")
            region: Geographic region (e.g., "India", "Southeast Asia")

        Returns:
            Compiled search results
        """

        queries = [
            f"{disease_name} prevalence {region}",
            f"{disease_name} outbreak {region} 2025",
            f"common causes {disease_name} {region}",
        ]

        results = {}
        for query in queries:
            result = self._perform_search(query)
            results[query] = result

        return self._format_results(results)

    def search_regional_pathogens(self, region: str, symptom_type: str = None) -> str:
        """
        Search for common pathogens in a region

        Args:
            region: Geographic region
            symptom_type: Type of symptoms (e.g., "fever", "diarrhea", "respiratory")

        Returns:
            List of common pathogens in that region
        """

        queries = [
            f"common pathogens {region}",
            f"endemic diseases {region}",
        ]

        if symptom_type:
            queries.append(f"{symptom_type} causing pathogens {region}")

        results = {}
        for query in queries:
            result = self._perform_search(query)
            results[query] = result

        return self._format_results(results)

    def search_food_safety_region(self, region: str, food_item: str = None) -> str:
        """
        Search for common foodborne pathogens in a region

        Args:
            region: Geographic region
            food_item: Specific food (e.g., "street food", "seafood")

        Returns:
            Foodborne pathogen information
        """

        queries = [
            f"foodborne illness {region}",
            f"common food poisoning {region}",
        ]

        if food_item:
            queries.append(f"{food_item} safety {region}")

        results = {}
        for query in queries:
            result = self._perform_search(query)
            results[query] = result

        return self._format_results(results)

    def search_vector_borne_diseases(self, region: str) -> str:
        """
        Search for tick/mosquito-borne diseases in a region

        Args:
            region: Geographic region

        Returns:
            Vector-borne disease information
        """

        queries = [
            f"tick-borne diseases {region}",
            f"mosquito-borne diseases {region}",
            f"endemic parasites {region}",
        ]

        results = {}
        for query in queries:
            result = self._perform_search(query)
            results[query] = result

        return self._format_results(results)

    def _perform_search(self, query: str) -> str:
        """
        Perform actual web search (using simple method)

        Args:
            query: Search query

        Returns:
            Search results
        """

        try:
            # Using DuckDuckGo for simple searches (no API key needed)
            from duckduckgo_search import DDGS

            ddgs = DDGS()
            results = list(ddgs.text(query, max_results=3))

            formatted = []
            for r in results:
                formatted.append(f"• {r['title']}: {r['body'][:200]}...")

            return "\n".join(formatted) if formatted else "No results found"

        except Exception as e:
            return f"Search error: {str(e)}"

    def _format_results(self, results: Dict) -> str:
        """Format search results nicely"""

        formatted = "ONLINE SEARCH RESULTS:\n"
        formatted += "=" * 60 + "\n\n"

        for query, result in results.items():
            formatted += f"Query: {query}\n"
            formatted += result + "\n\n"

        return formatted

    def synthesize_epidemiological_context(self, symptoms: str, region: str) -> str:
        """
        Use LLM to synthesize search results into clinical context

        Args:
            symptoms: Patient symptoms
            region: Patient's geographic region

        Returns:
            Clinical context based on regional epidemiology
        """

        # Get pathogen info for region
        pathogens = self.search_regional_pathogens(region)

        prompt_template = ChatPromptTemplate.from_template("""
You are a clinical epidemiologist. Based on this patient's symptoms and their geographic region, 
synthesize the epidemiological context.

PATIENT SYMPTOMS: {symptoms}
PATIENT REGION: {region}

REGIONAL EPIDEMIOLOGICAL DATA:
{epidemiological_data}

TASK: Provide a brief clinical context:
1. What diseases are ENDEMIC in this region?
2. What pathogens are COMMON in this region?
3. How do these align with the patient's symptoms?
4. What is the LIKELIHOOD of each disease in this region?

Format as a concise clinical summary (3-4 sentences).
""")

        chain = prompt_template | self.model
        context = chain.invoke({
            "symptoms": symptoms,
            "region": region,
            "epidemiological_data": pathogens
        })

        return context
