import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(layout="wide")
df=pd.read_csv("2019.csv")
button=st.button("Mapping Global Happiness: What the Data Tells Us About Joy Around the World")
if button:
  st.title("World Happiness Dashboard")
  st.subheader("Mapping Happiness : What makes countries happy?")
  st.html("""<h2><b>In a world constantly evolving with technological progress and global challenges, one question remains timeless:</b></h2>

<h3><i>What truly makes people happy?</i></h3>

<p>
Each year, the <strong>World Happiness Report</strong> attempts to answer this question by ranking countries based on how happy their citizens feel — and more importantly, <strong>why</strong> they feel that way. Backed by data from Gallup surveys and indicators like <strong>GDP per capita</strong>, <strong>social support</strong>, and <strong>freedom to make life choices</strong>, this report offers a unique lens into global well-being.
</p>

<p>
In this blog, we'll explore the <strong>World Happiness Report dataset</strong> through beautiful visualizations and data-driven insights. We'll dive into:
</p>

<ul>
  <li>🌍 <strong>How happiness is distributed across the globe</strong></li>
  <li>📈 <strong>What factors correlate most with happiness</strong></li>
  <li>🏆 <strong>Which countries consistently top the charts — and why</strong></li>
  <li>🔍 <strong>Surprising trends that challenge assumptions like "Money buys happiness"</strong></li>
</ul>

<p>
Whether you're a data enthusiast, a global thinker, or just curious to know where your country stands — <strong>this visual journey is for you.</strong>
</p>
<h2><b>📊 Visualizing the Pursuit of Happiness</b></h2>

<p>
To truly understand what drives happiness across the world, we’ll use powerful visualizations built with <strong>Seaborn</strong> and <strong>Plotly</strong>. These plots help reveal patterns, outliers, and surprising stories hidden in the data.
</p>
<h2><b>🧠 Understanding the Plots</b></h2>

<h3>📍 Scatter Plot</h3>
<p>
Each point represents a country. The x-axis and y-axis show relationships between two continuous variables, like <b>GDP per capita vs Happiness Score</b>. 
The color often distinguishes countries, and the size represents the magnitude of happiness.
</p>
<p><b>Insights:</b> As GDP or Freedom increases, Happiness Score often rises — showing positive correlation.</p>

<h3>📊 Bar Plot</h3>
<p>
Bar plots compare happiness scores across top-ranked countries. They are color-coded by factors such as GDP or life expectancy to explore their influence on happiness.
</p>
<p><b>Insights:</b> Countries like Finland and Denmark often top the list, with high values in key happiness indicators.</p>

<h3>🌡️ Correlation Heatmap</h3>
<p>
This plot shows how strongly different variables relate to one another. A darker color indicates a stronger positive relationship.
</p>
<p><b>Insights:</b> GDP, Social Support, and Freedom show strong positive correlations with the Happiness Score.</p>

<h3>🌍 Choropleth Plot (World Map)</h3>
<p>
Displays Happiness Score geographically. Countries are colored by happiness levels — darker means happier.
</p>
<p><b>Insights:</b> Northern Europe shows high scores, while some lower-income regions score lower.</p>

<h3>📦 Box Plot</h3>
<p>
Box plots summarize the distribution of happiness scores across countries. The box shows the interquartile range, the line shows the median, and dots represent outliers.
</p>
<p><b>Insights:</b> Helps compare variability in happiness across different regions or groups.</p>

<h3>🎻 Violin Plot</h3>
<p>
Violin plots combine box plots and KDE (density). They show both spread and concentration of scores.
</p>
<p><b>Insights:</b> Wider sections indicate where most countries fall in the happiness distribution.</p>

<h3>🧱 Stacked Bar Plot</h3>
<p>
Breaks down the total happiness score by contributing factors like GDP, Health, and Freedom.
</p>
<p><b>Insights:</b> Shows which factor contributes most in each country — useful for country-level comparison.</p>""")
st.sidebar.title("Various plots")
line=st.sidebar.button("Scatter plot")
bar=st.sidebar.button("Bar plot for top 10 countries")
heatmap=st.sidebar.button("Heatmap")
choropleth=st.sidebar.button("Choropleth plot")
stacked_bar=st.sidebar.button("Stacked bar plot")
boxplot=st.sidebar.button("Boxplot")
if line:
    st.html("<h2><b>GDP per capita vs Score</b></h2>")
    fig=px.scatter(df,x="GDP per capita",y="Score",color="Country or region",size="Score")
    st.plotly_chart(fig,use_container_width=True)
    st.text("""In the scatter plot above, each point represents a country. The x-axis shows the GDP per capita, while the y-axis shows the Happiness Score. Additionally:

Color represents the country's name for identification.

Size reflects the magnitude of the happiness score.

🟢 Observation:
There is a clear positive correlation — as GDP per capita increases, the happiness score tends to increase as well. This indicates that countries with stronger economies generally report higher levels of happiness.

However, while this upward trend is noticeable, the plot also reveals that GDP alone doesn't fully explain happiness. There are countries with similar GDP values but differing happiness scores, hinting at the influence of other factors like social support, freedom, and trust in government.

""")
    st.html("<h2><b>Healthy life expectancy vs Score</b></h2>")
    fig=px.scatter(df,x="Healthy life expectancy",y="Score",color="Country or region",size="Score")
    st.plotly_chart(fig,use_container_width=True)
    st.text("""In the scatter plot above, a similar trend is observed between GDP per capita and Happiness Score — as GDP increases, the Happiness Score also tends to rise, suggesting a positive and proportionate relationship.""")
    st.html("<h2><b>Freedom to make life choices vs Score</b></h2>")
    fig=px.scatter(df,x="Freedom to make life choices",y="Score",color="Country or region",size="Score")
    st.plotly_chart(fig,use_container_width=True)
    st.text("""The scatter plot above illustrates a strong positive relationship between freedom to make life choices and overall happiness. Countries that empower their citizens to make decisions about their lives tend to report higher happiness scores, highlighting the importance of autonomy in well-being.
While the trend is positive, there are still countries with moderate freedom scores but lower happiness, which could be influenced by other factors like GDP, social support, or health.""")

if bar:
    df2=df.sort_values(by="Score",ascending=False).head(10)
    st.html("<h2><b>Top 10 countries vs Score</b></h2>")
    st.html("<h3><b>Colour based on gdp per capita</b></h3>")
    fig=px.bar(df2,x="Country or region",y="Score",color="GDP per capita")
    st.plotly_chart(fig,use_container_width=True)
    st.html("<h3><b>Colour based on healthy life expectancy")
    fig=px.bar(df2,x="Country or region",y="Score",color="Healthy life expectancy")
    st.plotly_chart(fig,use_container_width=True)
    st.html("<h3><b>Colour based on freedom to make life choices</b></h3>")
    fig=px.bar(df2,x="Country or region",y="Score",color="Freedom to make life choices")
    st.plotly_chart(fig,use_container_width=True)

if heatmap:
    df2=df[["GDP per capita","Social support","Freedom to make life choices","Score"]].corr()
    st.html("<h2><b>Heatmap</b></h2>")
    fig=px.imshow(df2)
    st.plotly_chart(fig)
    st.html("""<h3><b>🧩 Heatmap Interpretation</b></h3>

<p>
The heatmap above illustrates the <strong>correlation</strong> between key happiness factors — <strong>GDP per capita</strong>, <strong>Social Support</strong>, <strong>Freedom to make life choices</strong>, and the overall <strong>Happiness Score</strong>.
</p>

<ul>
  <li><strong>Positive Correlations:</strong>
    <ul>
      <li>All three factors show a <strong>strong positive correlation</strong> with the <strong>Happiness Score</strong>.</li>
      <li>This suggests that countries with <em>higher income levels</em>, <em>stronger community support</em>, and <em>greater personal freedom</em> tend to report <strong>higher levels of happiness</strong>.</li>
    </ul>
  </li>

  <li><strong>GDP per capita:</strong>
    <ul>
      <li>Positively correlated with both <strong>Score</strong> and <strong>Social Support</strong>, indicating that wealthier nations may also be better equipped to support their citizens.</li>
    </ul>
  </li>

  <li><strong>Freedom to make life choices:</strong>
    <ul>
      <li>Shows strong alignment with <strong>Score</strong>, reinforcing the idea that autonomy plays a major role in well-being.</li>
    </ul>
  </li>
</ul>

<p>
Together, these insights emphasize that happiness is not driven by just one factor — rather, it's a combination of <strong>economic stability</strong>, <strong>social connections</strong>, and <strong>freedom</strong> that leads to happier societies.
</p>""")

if choropleth:
    st.html("<h2><b>Choropleth plot</b></h2>")
    fig=px.choropleth(df,locations="Country or region",locationmode="country names",color="Score",color_continuous_scale="viridis")
    st.plotly_chart(fig,use_container_width=True)
    st.html("""<h3><b>🌍 Global Happiness Visualization</b></h3>

<p>
The plot above shows each country represented as a point on the map or graph, where the <strong>color</strong> of each point indicates the country's <strong>Happiness Score</strong>. Brighter or more saturated colors typically represent <strong>higher scores</strong>, while muted tones indicate <strong>lower happiness levels</strong>.
</p>

<p>
This visual approach allows us to quickly identify <strong>regional patterns</strong>, spot <strong>outliers</strong>, and understand the global distribution of well-being. For instance, countries in Northern and Western Europe often appear with high scores, while some developing regions may reflect lower happiness levels.
</p>

<p>
Such plots are essential in <strong>making cross-country comparisons</strong> intuitive and visually compelling.
</p>""")

if stacked_bar:
        st.html("<h2><b>Stacked Bar Plot of Factors for Top 10 Countries</b></h2>")

        # Select top 10 countries by Score
        top_countries = df.sort_values("Score", ascending=False).head(10)

        # Prepare data
        data = top_countries[
            ["Country or region", "GDP per capita", "Healthy life expectancy", "Freedom to make life choices"]]
        data = data.set_index("Country or region")

        # Transpose to make countries columns
        data = data.T

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        data.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("Top 10 Countries - Contribution of Factors to Happiness")
        ax.set_ylabel("Score Contribution")
        ax.set_xlabel("Factors")
        plt.xticks(rotation=0)

        st.pyplot(fig)

if boxplot:
    st.html("<h2><b>Box plot</b></h2>")
    fig=px.box(df,x="Score")
    st.plotly_chart(fig,use_container_width=True)
    st.html("<h3>📦 The above boxplot illustrates the distribution of Happiness Scores across countries. It highlights the median, quartiles, and potential outliers — offering insight into how happiness varies globally.</h3>")
    st.html("<h2><b>Violin plot</b></h2>")
    fig=px.violin(df,x="Score")
    st.plotly_chart(fig,use_container_width=True)
    st.html("<h3>🎻 The above violin plot visualizes the distribution of Happiness Scores across countries. It combines a boxplot with a rotated kernel density plot, providing a richer view of the data’s spread, central values, and frequency distribution.</h3>")


