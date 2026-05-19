from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from ai_based_music_recommendation.tools import (
    SongLookupTool,
    SimilarSongSearchTool,
    SHAPExplainerTool,
)


@CrewBase
class MusicRecommendationCrew:
    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def song_analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config["song_analyzer"],
            tools=[SongLookupTool()],
            verbose=True,
        )

    @agent
    def music_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["music_researcher"],
            tools=[SimilarSongSearchTool()],
            verbose=True,
        )

    @agent
    def explainability_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["explainability_agent"],
            tools=[SHAPExplainerTool()],
            verbose=True,
        )

    @task
    def analyze_song_task(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_song_task"],
        )

    @task
    def research_similar_songs_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_similar_songs_task"],
        )

    @task
    def explain_and_recommend_task(self) -> Task:
        return Task(
            config=self.tasks_config["explain_and_recommend_task"],
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
