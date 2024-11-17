from parser.parser import parse_job, get_key_words
from processing.processing import flatten, merge_entries, process_dataframe
from model.summarize import summarize

import plotly.express as px
import streamlit as st

import asyncio
import pandas as pd
import numpy as np


async def main():
    st.title(":male-technologist: Анали вакансий hh.ru")

    st.header("Поиск вакансий", divider=True)
    st.markdown("#### Возможности анализатора")
    st.markdown("- :bar_chart: **Интерактивная инфографика**: легко визуализируйте рынок труда")
    st.markdown("- :briefcase: **Выделение ключевых навыков и требований**: узнайте, что ценится в вашей отрасли")
    st.markdown("- :hammer_and_wrench: **Расширенный поиск**: добавьте больше деталей, чтобы сузить круг поиска вакансий")
    st.markdown("Если хотите использовать расширенный поиск, отметьте галочку ниже. Тут можно узнать подробнее про [язык запросов](https://hh.ru/article/25295).")

    advanced_search = st.checkbox("Использовать расширенный поиск") 
    placeholder = '("Плейбой" OR "Миллиардер") NOT "Филлантроп"' if advanced_search else 'Плейбой, Миллиардер, Филлантроп'
    jobs = st.text_input('Тут должен быть запрос...', placeholder=placeholder)

    find_job_button = st.button("Найти вакансии")

    if find_job_button:
        job_query = jobs
        if not advanced_search:
            jobs = list(map(lambda x: x.strip(), jobs.split(',')))
            job_query = get_key_words(jobs)
        
        with st.spinner("Выполняется поиск вакансий…"):
            data = await parse_job(job_query, number_of_pages=100)
            flatten_data = list(map(flatten, data))
            merged_entries = merge_entries(flatten_data)
            df = pd.DataFrame(merged_entries)
            df = process_dataframe(df.copy())
            
        st.markdown(f":white_check_mark: Было найдено {df.shape[0]} вакансий!")

        st.header("Вакансии на карте", divider=True)
        st.map(df, latitude="address_lat", longitude="address_lng")

        st.header("Профессия в графиках", divider=True)
        
        st.markdown("### График работы")
        fig = px.pie(df, names='schedule_name', title='Распределение вакансий по графику работы')
        st.plotly_chart(fig)

        st.markdown("### Опыт работы")
        fig = px.pie(df, names='experience_name', title='Распределение вакансий по требуемому опыту работы')
        st.plotly_chart(fig)

        st.markdown("### Занятость")
        fig = px.pie(df, names='employment_name', title='Распределение вакансий по занятости')
        st.plotly_chart(fig)
    
        st.markdown("### Указана ли зарплата?")
        fig = px.histogram(df, x='is_salary_set')
        fig.update_layout(xaxis_title='Указана ли зарплата?', yaxis_title='Количество вакансий')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Распределение зарплат")
        
        salary_df = df[df['salary'] > 0]

        st.markdown("**Общее распределение зарплат**")
        mid_salary = int(np.median(salary_df['salary']))
        mean_salary = int(np.mean(salary_df['salary']))
        st.markdown(f"- Медианная зарплата для этой професии составляет: {mid_salary} рублей.")
        st.markdown(f"- Средняя зарплата для этой професии составляет: {mean_salary} рублей.")
        
        fig = px.histogram(salary_df, x='salary')
        fig.update_layout(xaxis_title='Зарплата', yaxis_title='Количество вакансий')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Распределение зарплат по опыту работы**")
        for skill_level in df['experience_name'].unique():
            skill_level_data = df[(df['experience_name'] == skill_level) & (df['salary'] > 0)]
            if not skill_level_data.empty:
                mid_salary = int(np.median(skill_level_data['salary']))
                mean_salary = int(np.mean(skill_level_data['salary']))
                st.markdown(f"**{skill_level}**:")
                st.markdown(f"- Медианная зарплата для этой профессии составляет: {mid_salary} рублей.")
                st.markdown(f"- Средняя зарплата для этой профессии составляет: {mean_salary} рублей.")

        experience_uniques = salary_df['experience_name'].unique()
        fig = px.histogram(df[df['salary'] > 0], x='salary', color='experience_name', title='Распределение зарплаты')
        fig.update_layout(xaxis_title='Зарплата', yaxis_title='Количество вакансий', legend_title="Опыт работы")
        fig.update_xaxes(categoryorder='array', categoryarray=experience_uniques)
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.box(salary_df, y='salary', x='experience_name', title="Разброс зарплаты в зависимости от опыта работы")
        fig.update_layout(xaxis_title='Опыт работы', yaxis_title='Зарплата')
        fig.update_xaxes(categoryorder='array', categoryarray=experience_uniques)
        st.plotly_chart(fig, use_container_width=True)

        st.header("Навыки и обязанности", divider=True)

        st.markdown("### Необходимые навыки")
        with st.spinner("Собираем данные по навыкам…"):
            for skill_level in df['experience_name'].unique():
                requirements = df[df['experience_name'] == skill_level]['snippet_requirement'].dropna()
                st.write(f"{skill_level}:")
                if len(requirements) == 0:
                    st.markdown("\tНет данных =(")
                    continue
                else:
                    requirements_summary = summarize(requirements, n_clusters=3)
                    for requirement in requirements_summary:
                        st.markdown(f"- {requirement}") 

        st.markdown("### Обязанности")
        with st.spinner("Собираем данные по обязанностям…"):
            for skill_level in df['experience_name'].unique():
                st.write(f"{skill_level}:")
                responsibilities = df[df['experience_name'] == skill_level]['snippet_responsibility'].dropna()
                if len(responsibilities) == 0:
                    st.markdown("\tНет данных =(")
                    continue
                else:
                    responsibility_summary = summarize(responsibilities, n_clusters=3)
                    for responsibility in responsibility_summary:
                        st.markdown(f"- {responsibility}") 

if __name__ == "__main__":
    asyncio.run(main()) 