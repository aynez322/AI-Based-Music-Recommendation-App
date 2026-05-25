from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from ai_based_music_recommendation.tools import (
    SongLookupTool,
    GenreFingerprinterTool,
    SimilarSongSearchTool,
    SHAPExplainerTool,
)


# definitia crew-ului de recomandare muzicala cu toti agentii si task-urile
@CrewBase
class MusicRecommendationCrew:
    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # agentul care analizeaza cantecul de intrare si ii extrage caracteristicile
    @agent
    def song_analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config["song_analyzer"],
            tools=[SongLookupTool()],
            verbose=True,
        )

    # agentul care cauta cantece similare folosind amprente de gen si similaritate cosinus
    @agent
    def music_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["music_researcher"],
            tools=[GenreFingerprinterTool(), SimilarSongSearchTool()],
            verbose=True,
        )

    # agentul care explica de ce cantecele recomandate sunt similare cu cel de intrare
    @agent
    def explainability_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["explainability_agent"],
            tools=[SHAPExplainerTool()],
            verbose=True,
        )

    # task-ul in care agentul analizeaza cantecul si ii identifica genul
    @task
    def analyze_song_task(self) -> Task:
        return Task(config=self.tasks_config["analyze_song_task"])

    # task-ul in care agentul cauta cantece similare in dataset
    @task
    def research_similar_songs_task(self) -> Task:
        return Task(config=self.tasks_config["research_similar_songs_task"])

    # task-ul in care agentul genereaza explicatii shap si produce raportul final
    @task
    def explain_and_recommend_task(self) -> Task:
        return Task(config=self.tasks_config["explain_and_recommend_task"])

    # creeaza crew-ul cu procesare secventiala a task-urilor
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
