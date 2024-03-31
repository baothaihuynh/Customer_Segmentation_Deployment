import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import streamlit as st
from PIL import Image
from wordcloud import WordCloud
import plotly.express as px
from plotly.graph_objects import Figure
from plotly.graph_objs import graph_objs


# Read data raw
df_1 = pd.read_csv("data_raw/data_raw_part1.csv")
df_2 = pd.read_csv("data_raw/data_raw_part1.csv")
df_3 = pd.read_csv("data_raw/data_raw_part1.csv")
data = pd.concat([df_1, df_2, df_3], axis=0).reset_index(drop=True)

# Read data cleaned
df_m1 = pd.read_csv("data_analysis_model_part/data_analysis_model_part1.csv")
df_m2 = pd.read_csv("data_analysis_model_part/data_analysis_model_part2.csv")
df_m3 = pd.read_csv("data_analysis_model_part/data_analysis_model_part3.csv")
data_clean = pd.concat([df_m1, df_m2, df_m3], axis=0)
data_clean["InvoiceDate"] = pd.to_datetime(data_clean["InvoiceDate"]).reset_index(
    drop=True
)
data_clean.loc[data_clean["Year Month"] == "2011-12", "Year Month"] = "2010-12"

rfm_agg = pd.read_csv("data_rfm/rfm_agg.csv")
retention = pd.read_csv("data_rfm/retention.csv")
rfm_data = pd.read_csv("data_rfm/data_rfm.csv")

# Read data final
df_f1 = pd.read_csv("data_final_part/data_final_part1.csv")
df_f2 = pd.read_csv("data_final_part/data_final_part2.csv")
df_f3 = pd.read_csv("data_final_part/data_final_part3.csv")
data_final = pd.concat([df_f1, df_f2, df_f3], axis=0).reset_index(drop=True)


# Read dataset not clean
# data = pd.read_csv("data/OnlineRetail.csv", encoding="ISO-8859-1")
# Read datasets cleaned
# data_clean = pd.read_csv("data_cleaned/data_analysis_model.csv")
# data_clean["InvoiceDate"] = pd.to_datetime(data_clean["InvoiceDate"])
# data_clean.loc[data_clean["Year Month"] == "2011-12", "Year Month"] = "2010-12"
# rfm_agg = pd.read_csv("data_cleaned/rfm_agg.csv")
# retention = pd.read_csv("data_cleaned/retention.csv")
# rfm_data = pd.read_csv("data_cleaned/data_rfm.csv")
# data_final = pd.read_csv("data_cleaned/data_final.csv")

# Task 1:
revenue_month = data_clean.groupby("Year Month")["Revenue"].sum().reset_index()
revenue_month["Revenue ($1000)"] = round((revenue_month["Revenue"] / 1000), 2)

# Task 3:
act_cus = data_clean.groupby("Year Month")["CustomerID"].nunique().reset_index()
act_cus = act_cus.rename(columns={"CustomerID": "Active Customers"})

# Task 5:
cus_in_county = data_clean.groupby(["Country"])["CustomerID"].nunique().reset_index()
cus_in_county.sort_values(by="CustomerID", ascending=False, inplace=True)
cus_in_county["Percent Customer"] = round(
    (cus_in_county["CustomerID"] / sum(cus_in_county["CustomerID"])), 2
)
cus_in_county = cus_in_county.rename(columns={"CustomerID": "Number Customers"})


# Build function for new predict
def join_rfm(x):
    return str(int(x["R"])) + str(int(x["F"])) + str(int(x["M"]))


def rfm_cluster(df):
    if df["R"] == 4 and df["F"] == 4 and df["M"] == 4:
        return "Best Customers"
    elif df["R"] == 4 and df["F"] == 1 and df["M"] == 1:
        return "New Customers"
    elif df["R"] == 1 and df["F"] in [1, 2] and df["M"] in [1, 2]:
        return "Lost Customers"
    elif (df["R"] in [3, 4]) and (df["F"] in [4]):
        return "Loyal Customers"
    elif df["M"] in [3, 4]:
        return "Big Spenders Customeres"
    else:
        return "Need Attention Customers"


def segment(df, rfm_data):
    # rfm_data = pd.read_csv("data_cleaned/data_rfm.csv")
    rfm_data_new = rfm_data.drop("CustomerID", axis=1)
    rfm_data_new = pd.concat([rfm_data_new, df]).reset_index(drop=True)
    r_range = range(4, 0, -1)
    f_range = range(1, 5)
    m_range = range(1, 5)
    # Assign these labels to 4 equal percentile groups
    R = pd.qcut(rfm_data_new["Recency"].rank(method="first"), q=4, labels=r_range)
    F = pd.qcut(rfm_data_new["Frequency"].rank(method="first"), q=4, labels=f_range)
    M = pd.qcut(rfm_data_new["Monetary"].rank(method="first"), q=4, labels=m_range)
    rfm_data_new = rfm_data_new.assign(R=R.values, F=F.values, M=M.values)
    rfm_data_new["RFM Score"] = rfm_data_new.apply(join_rfm, axis=1)
    rfm_data_new["Segment"] = rfm_data_new.apply(rfm_cluster, axis=1)
    rfm_data_new = rfm_data_new.assign(R=R.values, F=F.values, M=M.values)
    rfm_data_new["RFM Score"] = rfm_data_new.apply(join_rfm, axis=1)
    rfm_data_new["Segment"] = rfm_data_new.apply(rfm_cluster, axis=1)
    df_new = rfm_data_new[["Recency", "Frequency", "Monetary", "Segment"]].tail(
        len(rfm_data_new) - len(rfm_data)
    )
    return df_new


# Prepare text data
best_customer = """
 ###### Problem: How can businesses retain these customers?
   """

lost_customer = """
         - ###### Problem: What problems are they facing and how can businesses attract them back to buy?  
"""

loyal_customer = """ 
- ###### Problem: How to increase their purchase frequency and increase the cart value?
"""

need_ac_customer = """
- ###### Problem: What makes them dissatisfied and not buy regularly?
"""

new_customer = """
- ###### Problem: How to convert them from new customers to loyal customers of the business?
"""
big_spend_customer = """
- ###### Problem: How can we get them to buy more often?  
"""
# Create image Best Customer
bestcus = Image.open("image/recomment_bestcus.png")
new_width = 1000
new_height = 400
bestcus = bestcus.resize((new_width, new_height))
# st.image(img)


# Create image Big Spenders Customer
bigcus = Image.open("image/recomment_bigcus.png")
new_width = 1000
new_height = 400
bigcus = bigcus.resize((new_width, new_height))


# Create image Loyal Customer
loyalcus = Image.open("image/recomment_loyalcus.png")
new_width = 1000
new_height = 400
loyalcus = loyalcus.resize((new_width, new_height))


# Create image Need Attention Customer
needcus = Image.open("image/recomment_needcus.png")
new_width = 1000
new_height = 400
needcus = needcus.resize((new_width, new_height))

# Create image New Customer
newcus = Image.open("image/recomment_newcus.png")
new_width = 1000
new_height = 400
newcus = newcus.resize((new_width, new_height))

# Create image Lost Customer
lostcus = Image.open("image/recomment_lostcus.png")
new_width = 1000
new_height = 400
lostcus = lostcus.resize((new_width, new_height))


# Create title
st.title("DATA SCIENCE PROJECT")
st.write("## Customer Segmentation📚")

# Create image
img = Image.open("image/segment1.png")
new_width = 800
new_height = 400
img = img.resize((new_width, new_height))
st.image(img)

# Create menu
menu = [
    "📝Overview",
    "📊About Project",
    "🔎New Predict",
    "📂Find Customer Information in Dataset",
]
choice = st.sidebar.selectbox("TABLE OF CONTENTS", menu)

# Create Overview table
if choice == "📝Overview":
    st.subheader("📝Overview")

    # Business Objective
    st.write("#### I. Business Objective")
    st.write(
        """##### Building a customer segmentation system based on information provided by the company can help the company identify different customer groups in order to develop appropriate business and customer care strategies.
             """
    )

    # What is RFM in Customer Segnmentation?
    st.write("#### II. What is RFM in Customer Segnmentation?")

    # Create image
    img = Image.open("image/rfm.png")
    new_width = 800
    new_height = 400
    img = img.resize((new_width, new_height))
    st.image(img)

    st.write(
        """ ##### RFM stands for Recency, Frequency, and Monetary Value, and it is a technique used in marketing and customer segmentation to analyze and categorize customers based on their transaction behavior. Each of the three components has a specific meaning:
         """
    )

    st.write(
        """   
             - ##### Recency (R): How recently did the customer make a purchase?  
             - ##### Frequency (F): How often does the customer make purchases within a specific timeframe?  
             - ##### Monetary (M): How much money has the customer spent within a specific timeframe?
     """
    )
    st.write(
        """ ##### RFM Customer Segmentation helps businesses better understand their customers, target specific segments with tailored marketing efforts, enhance customer loyalty, and increase profitability through optimized marketing strategies.
    """
    )

    # About Dataset
    st.write("#### III. About Dataset")
    # Create image
    img = Image.open("image/online_retails.jpg")
    new_width = 800
    new_height = 400
    img = img.resize((new_width, new_height))
    st.image(img)
    st.write(
        "##### Link data: [Click Here](https://archive.ics.uci.edu/dataset/352/online+retail)"
    )

    # Team
    st.write(
        """
    #### IV. Team
    - ##### Huỳnh Thái Bảo  
    - ##### Đặng Lê Hoàng Tuấn   
    """
    )


elif choice == "📊About Project":
    st.subheader("📊About Project")

    # Summary Dataset
    st.write("#### I. Summary Dataset")
    st.dataframe(data.head(3))
    st.dataframe(data.tail(3))

    # Exploratory Data Analysis
    st.write("#### II. Exploratory Data Analysis")

    # Task 1: What is the monthly revenue?
    st.write(" ##### 1. What is the monthly revenue?")

    months = [x for x in revenue_month["Year Month"]]
    # Assuming your data is in a pandas dataframe called 'revenue_month'
    # Create a list of colors
    colors = ["#07B3B2"] * 9 + ["#005A74"] * 3
    # Create a plotly figure object
    fig = Figure()
    # Add bars to the figure
    fig.add_traces(
        graph_objs.Bar(
            x=revenue_month["Year Month"],
            y=revenue_month["Revenue ($1000)"],
            marker_color=colors,
        )
    )
    # Set title and layout options
    fig.update_layout(
        title="Monthly Revenue",
        title_x=0,  # Left align title
        title_font_size=30,
        yaxis_title="Revenue ($1000)",
        xaxis_ticktext=months,
        xaxis_tickformat="%m-%Y",
        height=600,
        width=900,
    )
    # Show the plotly chart
    st.plotly_chart(fig)

    st.write(""" ##### Conclusion: """)
    st.write(
        """
             ###### With an average revenue of 781,999 USD/month, we can have a general overview of the company's retail situation as follows:
             ###### 1. Revenue in the first half of the year:
             - ###### January to August: Revenue remained stable at below 800,000 USD/month.
             ###### 2. Revenue in the second half of the year:  
             - ###### September to November: Revenue showed a significant increase, exceeding 1,000,000 USD per month.  
             - ###### November: Revenue peaked at over 1,400,000 USD.   
             ###### Prediction:
             - ###### There is a trend of increasing purchases at the end of the year.     
             ###### Recommendation:  
             - ###### Analyze data on order volume and products sold to reinforce the prediction. 
             """
    )

    # Task 2: Customer Shopping Trend Analysis
    st.write(" ##### 2. Customer Shopping Trend Analysis")
    # Create image
    img = Image.open("image/4kpi.png")
    new_width = 3500
    new_height = 2500
    img = img.resize((new_width, new_height))
    st.image(img)
    st.write(""" ##### Conclusion: """)
    st.write(
        """
             ###### Based on the four KPIs analyzed, we can confirm that the hypothesis in Task 1 is valid.  
             ###### Three out of four indicators show a strong growth trend from September, peaking in November.   
             ###### It is noticeable that these three months have many major holidays that impact purchasing power through various forms of discounts, promotions, and gifts. Some notable days include:  
             - ###### Labor Day (First Monday of September) in the US.  
             - ###### Halloween (October 31st) => Increased retail sales for costumes, candy, decorations, and Halloween-related products.  
             - ###### Thanksgiving (4th Thursday of November) => Increased retail sales for food, kitchenware, and decorations.  
             - ###### Black Friday (4th Friday of November) => This is the biggest discount day of the year, boosting sales for businesses across all categories.      
    """
    )
    st.write(""" ##### Recommendations: """)
    st.write(
        """
             ###### Implement various promotional programs, discounts, gifts, etc., throughout the year to attract continuous customer shopping instead of focusing only on holidays and special occasions.  
             ###### Analyze the trends of products purchased by customers in the 3 peak months of the year to promote diversified business with similar products.   
             ###### Connect more with customers, collect valuable feedback and reviews. From there, improve and overcome limitations and maximize the strengths of the business.  
             ###### Analyze overall market trends to constantly update products, prices, and policies so that the business always creates a fresh feeling for customers, attracting them at all times of the year.
    """
    )

    # Task 3: Monthly Active Customers (MAC)
    st.write(" ##### 3. Monthly Active Customers (MAC)?")
    colors = ["#07B3B2"] * 9 + ["#005A74"] * 3
    # Create a plotly figure object
    fig = Figure()
    # Add bars to the figure
    fig.add_traces(
        graph_objs.Bar(
            x=act_cus["Year Month"],
            y=act_cus["Active Customers"],
            marker_color=colors,
        )
    )
    # Set title and layout options
    fig.update_layout(
        title="Monthly Active Customers",
        title_x=0,  # Left align title
        title_font_size=30,
        yaxis_title="Active Customers",
        xaxis_ticktext=months,
        xaxis_tickformat="%m-%Y",
        height=600,
        width=900,
    )
    st.plotly_chart(fig)

    st.write(""" ##### Conclusion: """)
    st.write(
        """
             ###### Following the growth trends analyzed above, the monthly active customer trend shows similar patterns.  
             ###### Specifically:  
             - ###### From September to November: The number of active customers increased rapidly and peaked in November with over 1,750 customers.  
             - ###### Meanwhile: The remaining months of the year maintained the number of active customers per month between 900 and 1,200.  
             ###### The number of active customers fluctuates seasonally, with November being the month with the highest number of active customers.  
             ###### This growth is consistent with the overall growth trend of revenue and other metrics.  
    """
    )

    # Task 4: Monthly Retention Rate (MRR)
    st.write(" ##### 4. Monthly Retention Rate (MRR)?")
    fig = Figure()
    # Add bars to the figure
    fig.add_traces(
        graph_objs.Line(
            x=retention["Year Month"],
            y=retention["Customers Retention Rate (%)"],
            marker_color="#005A74",
        )
    )
    # Set title and layout options
    fig.update_layout(
        title="Monthly Retention Rate",
        title_x=0,  # Left align title
        title_font_size=30,
        yaxis_title="Customers Retention Rate (%)",
        xaxis_ticktext=months,
        xaxis_tickformat="%m-%Y",
        height=600,
        width=900,
    )
    st.plotly_chart(fig)

    st.write(""" ##### Conclusion: """)
    st.write(
        """
             ###### The data shows that the company's Monthly Retention Rate (MRR) has an upward trend over each month.  
             ###### However:  
             - ###### There is a decrease in MRR in March.
             - ###### From April to the end of the year: MRR recovers and maintains a stable level.  
             ###### Overall, the company's MRR has a positive and stable trend throughout the year.  
             ###### The decrease in March could be due to factors such as:  
             - ###### Changes in business strategy.  
             - ###### New product launch.  
             - ###### Increased competition.    
    """
    )

    # Task 5: How many customers are there in each region?
    st.write(" ##### 5. How many customers are there in each region?")
    # Assuming your data is in a pandas datafr
    # Create a list of colors
    colors = ["#005A74"] * 1 + ["#07B3B2"] * (len(cus_in_county) - 1)
    # Create a plotly figure object
    fig = Figure()
    # Add bars to the figure
    fig.add_traces(
        graph_objs.Bar(
            x=cus_in_county["Country"],
            y=cus_in_county["Number Customers"],
            marker_color=colors,
        )
    )
    # Set title and layout options
    fig.update_layout(
        title="Number Customers for each County",
        title_x=0,  # Left align title
        title_font_size=30,
        yaxis_title="Number Customers",
        height=600,
        width=900,
    )
    st.plotly_chart(fig)

    st.write(""" ##### Conclusion: """)
    st.write(
        """
             ###### The data shows that the company's main customer base is mainly from the UNITED KINGDOM.  
             ###### This indicates that:  
             - ###### The company's current main market is the United Kingdom.  
             - ###### The company needs to analyze customer data from other regions more carefully to expand its market and diversify its revenue streams.  
             ###### To expand its market, the company can:  
             - ###### Analyze customer data from other regions to better understand their needs and preferences.  
             - ###### Adjust products and services to meet the needs of the target market. 
             - ###### Implement marketing campaigns targeted at potential regions.  
             - ###### Cooperate with local partners to penetrate new markets.  
             ###### Expanding the market helps the company to:  
             - ###### Increase revenue from different regions.  
             - ###### Minimize the risk of relying on a single market.  
             - ###### Enhance competitive position in the market.  
    """
    )

    # Customer Segmentation
    st.write("#### III. Customer Segmentation")
    st.write("##### 1. Result")
    st.write("- ###### RFM Data")
    st.dataframe(rfm_agg)
    st.write("- ###### Visualize")
    fig = px.scatter(
        rfm_agg,
        x="Recency Mean",
        y="Monetary Mean",
        size="Frequency Mean",
        color="Segment",
        hover_name="Segment",
        size_max=100,
    )
    fig.update_layout(height=500, width=1000)
    st.plotly_chart(fig)
    st.write(
        "- ###### The results of the classification method using the rule set based on RFM data are relatively good."
    )
    st.write("- ###### The number of customers in each group is not too different.")
    st.write("##### 2. Conclusion")
    # Create image
    img = Image.open("image/segment_details.png")
    new_width = 3500
    new_height = 2500
    img = img.resize((new_width, new_height))
    st.image(img)

    # Best Customer
    st.write(
        """
             - ###### Best Customer  
                - ###### This is a group of customers who buy frequently with very high purchase frequency and value. This is a group of customers that businesses cannot afford to lose.
                - ###### Summary:  
                    - ###### Recency: The average time to the most recent purchase is 8 days. The time ranges mainly from 1 to 10 days. (Recent)  
                    - ###### Frequency: The average purchase frequency is 17 times per year. The frequency ranges from 10 to 30 times. (Very high)  
                    - ###### Monetary: The average transaction value is 7830. The value ranges from 1000 to 10000. (Very high)  
                - ###### Problem: How can businesses retain these customers?
                - ###### Recommendations:
                    - ###### Give them early access to new products: This group of customers tends to like to experience new things. Therefore, giving them early access to new products will help them feel valued and can lead to long-term loyalty to the business.  
                    - ###### Encourage them to participate in loyalty programs: Loyalty programs are an effective way to retain customers. Businesses can design programs with many attractive offers specifically for this group of customers, such as point redemption, discounts, free shipping,...  
                    - ###### Build promotional and discount programs specifically for them: This is a direct way to attract and retain customers. Businesses can design promotional programs that are tailored to the needs and interests of this group of customers.           
             """
    )

    # Big Spenders Customer
    st.write(
        """
             - ###### Big Spenders Cutomer
                 - ###### Customers with average purchase frequency but high order value. This group of customers purchases with average frequency but high order value. This group of customers brings high profits to the business and needs to be focused on.
                 - ###### Summary:  
                    - ###### Recency: The average time to the most recent purchase is 126 days. The time ranges mainly from 50 to 200 days. (Occasionally)
                    - ###### Frequency: The average purchase frequency is 2 times per year. The frequency ranges from 1 to 4 times. (Low)
                    - ###### Monetary: The average transaction value is 1727. The value ranges from 1000 to 3000. (High)  
                 - ###### Problem: How can we get them to buy more often?  
                 - ###### Recommendations:  
                    - ###### Encourage them to participate in VIP customer programs: VIP customer programs are an effective way to encourage customers to buy more often. Businesses can design programs with many attractive offers specifically for this group of customers, such as point redemption, discounts, free shipping,...  
                    - ###### Analyze their shopping cart history: Businesses can analyze the shopping cart history of this group of customers to identify the products they often buy. Then, businesses can offer discounts and buy-one-get-one-free programs for similar products.  
    """
    )

    # Lost Customer
    st.write(
        """
             - ###### Lost Customer  
                 - ###### Customers who haven't bought in a long time. This group of customers has only purchased once and has not purchased again for a long time. It can be determined that this is a group of customers that we have lost.
                 - ###### Summary:  
                    - ###### Recency: The average time to the most recent purchase is 289 days. The time ranges over 200 days. (Very long)
                    - ###### Frequency: The average purchase frequency is 1 time per year. (Very low)
                    - ###### Monetary: The average transaction value is 9. The value mostly fluctuates below 0 (group with return transactions). (Very low)
                 - ###### Problem: What problems are they facing and how can businesses attract them back to buy?  
                 - ###### Recommendations:  
                    - ###### Analyze their shopping cart history: Businesses can analyze the shopping cart history of this group of customers to identify the problems they are facing. For example: products do not meet needs, poor customer service,... From there, businesses can improve their weaknesses to attract them back to buy.
                    - ###### Reconnect with these customers: Businesses can reconnect with these customers through various channels such as email, direct social media interaction, or phone calls. The purpose is to listen and ask for their feedback on their shopping experiences at the business.
                    - ###### Implement retargeting campaigns: Businesses can implement retargeting campaigns to display ads for their products and services to this group of customers on social media channels or websites they often visit.
                    - ###### Implement short-term promotional programs: Businesses can implement short-term promotional programs with vouchers, discounts, exclusive offers,... to attract them back to buy.
                    - ###### Offer free trials: Businesses can offer free trials of their products and services to encourage them to return and purchase.
    """
    )

    # Loyal Customer
    st.write(
        """
             - ###### Loyal Customer  
                 - ###### This group of customers purchases with relatively high frequency and value. This is a potential customer group for businesses.  
                 - ###### Summary:  
                    - ###### Recency: The average time to the most recent purchase is 30 days. The time ranges mainly from 20 to 50 days. (Frequent)
                    - ###### Frequency: The average purchase frequency is 7 times per year. The frequency ranges from 5 to 10 times. (Quite high)
                    - ###### Monetary: The average transaction value is 2191. The value ranges from 1000 to 4000. (Quite high)
                 - ###### Problem: How to increase their purchase frequency and increase the cart value?
                 - ###### Recommendations:  
                    - ###### Upsell high-value products: Businesses can introduce and recommend high-value products to this group of customers. Effective upselling will help increase cart value and revenue for businesses.
                    - ###### Ask for their feedback: Businesses can conduct surveys, interviews, or chat directly with this group of customers to ask for their feedback on the business's products and services. From there, businesses can improve their products and services to better meet customer needs.
                    - ###### Implement customer engagement campaigns: Businesses can implement marketing campaigns to increase engagement with this group of customers. For example: loyalty programs, customer appreciation programs,...  
                    - ###### Give gifts when they make transactions above a certain value threshold: Businesses can give gifts to this group of customers when they make transactions above a certain value threshold. This is an effective way to encourage them to buy more.      
    """
    )

    # Need Attention Customer
    st.write(
        """
             - ###### Need Attention Customer
                 - ###### Customers who are facing problems. This group of customers has an average purchase frequency but is facing some problems that need to be solved, including many negative transactions (return transactions).
                 - ###### Summary:  
                    - ###### Recency: The average time to the most recent purchase is 98 days. The time ranges mainly from 50 to 200 days. (Average)
                    - ###### Frequency: The average purchase frequency is 2 times per year. The frequency ranges from 1 to 3 times. (Quite low)
                    - ###### Monetary: The average transaction value is 175. The value ranges from 100 to below 0. (Very low)
                 - ###### Problem: What makes them dissatisfied and not buy regularly?
                 - ###### Recommendations:
                    - ###### Offer special limited-time promotions: Businesses can offer special limited-time promotions to this group of customers to encourage them to buy more often.
                    - ###### Recommend based on products they have purchased: Businesses can analyze the purchase history of this group of customers and suggest similar products or new products that they may be interested in.
                    - ###### Give gifts when they make transactions above a certain value threshold: Businesses can give gifts to this group of customers when they make transactions above a certain value threshold. This is an effective way to encourage them to buy more.
                    - ###### Analyze their shopping cart history: Businesses can analyze the shopping cart history of this group of customers to identify any product-related factors that contribute to their dissatisfaction. For example: products do not meet needs, product quality is not good,... From there, businesses can improve their products to better meet customer needs.
    """
    )

    # New Customer
    st.write(
        """
             - ###### New Customer
                 - ###### This group of customers is new to the business and has not made many transactions yet, and the order value is still low.
                 - ###### Summary:  
                    - ###### Recency: The average time to the most recent purchase is 13 days. The time ranges mainly from 5 to 15 days. (Recent)
                    - ###### Frequency: The average purchase frequency is 1 time per year. (Very low)
                    - ###### Monetary: The average transaction value is 136. The value ranges from 100 to 150. (Very low)
                 - ###### Problem: How to convert them from new customers to loyal customers of the business?
                 - ###### Recommendations:
                    - ###### Encourage them to participate in new membership programs: Businesses can design new membership programs with many attractive offers specifically for this group of customers, such as point redemption, discounts, free shipping,...
                    - ###### Recommend based on products they have purchased: Businesses can analyze the purchase history of this group of customers and suggest similar products or new products that they may be interested in.
                    - ###### Conduct satisfaction surveys: Businesses can conduct satisfaction surveys of this group of customers after they purchase to collect feedback on their shopping experience. From there, businesses can promote their strengths and improve their weaknesses to attract them back to buy more often.                     
    """
    )


# Create New Predict
elif choice == "🔎New Predict":
    st.subheader("🔎New Predict")

    # Select data
    st.write("##### I. Select Data")
    flag = False
    lines = None
    type = st.radio(
        "Do you want to Input data or Upload data?", options=("Input", "Upload")
    )

    if type == "Input":
        data_new = pd.DataFrame(columns=["Recency", "Frequency", "Monetary"])
        select_num_cus = st.radio(
            "Do want to predict for 1 customer or 5 customers?",
            options=("One Customer", "Five Customers"),
        )
        # Select 1 Customer
        if select_num_cus == "One Customer":
            # Create sliders to input values for R, F, M column
            recency = st.slider("Recency", 1, 365, 100)
            frequency = st.slider("Frequency", 1, 50, 5)
            monetary = st.slider("Monetary", -10000, 10000, 100)
            # Append new data in data_new df
            new_data = pd.DataFrame(
                {"Recency": recency, "Frequency": frequency, "Monetary": monetary},
                index=[0],
            )
            data_new = pd.concat([data_new, new_data], ignore_index=True)
            df_predict = segment(data_new, rfm_data)
            df_predict = df_predict.reset_index(drop=True)
            submitted = st.button("Submit")
            if submitted:
                st.write("##### II. Result")
                segments = rfm_agg["Segment"].unique()
                for segment in segments:
                    # Filter df by segment
                    filtered_df = df_predict[df_predict["Segment"] == segment]

                    # Print df after filter
                    if not filtered_df.empty:
                        st.write(f"###### Segment: {segment}")
                        st.write(filtered_df)
                        if segment == "Best Customers":
                            st.write(best_customer)
                            st.image(bestcus)
                        elif segment == "Big Spenders Customers":
                            st.write(big_spend_customer)
                            st.image(bigcus)
                        elif segment == "Lost Customers":
                            st.write(lost_customer)
                            st.image(lostcus)
                        elif segment == "Loyal Customers":
                            st.write(loyal_customer)
                            st.image(loyalcus)
                        elif segment == "Need Attention Customers":
                            st.write(need_ac_customer)
                            st.image(needcus)
                        else:
                            st.write(new_customer)
                            st.image(newcus)
                csv = df_predict.to_csv(index=False)
                st.download_button(
                    label="Download predictions as CSV file",
                    data=csv,
                    file_name="predictions.csv",
                    mime="csv",
                )

        # Selecct 5 Customers
        elif select_num_cus == "Five Customers":
            recency = []
            frequency = []
            monetary = []
            # Create sliders to input values for R, F, M colum
            for i in range(5):
                st.write(f"- ##### Cusomter {i + 1}")
                r = st.slider("Recency", 1, 365, 100, key=f"Recency {i}")
                f = st.slider("Frequency", 1, 50, 5, key=f"Frequency {i}")
                m = st.slider("Monetary", -10000, 10000, 100, key=f"Monetary {i}")
                recency.append(r)
                frequency.append(f)
                monetary.append(m)
            # Append new data in data_new df
            new_data = pd.DataFrame(
                {"Recency": recency, "Frequency": frequency, "Monetary": monetary},
                index=range(5),
            )
            data_new = pd.concat([data_new, new_data], ignore_index=True)
            df_predict = segment(data_new, rfm_data)
            df_predict = df_predict.reset_index(drop=True)
            submitted = st.button("Submit")
            if submitted:
                st.write("##### II. Result")
                segments = rfm_agg["Segment"].unique()
                for segment in segments:
                    # Filter df by segment
                    filtered_df = df_predict[df_predict["Segment"] == segment]
                    # Print df after filter
                    if not filtered_df.empty:
                        st.write(f"###### Segment: {segment}")
                        st.write(filtered_df)
                        if segment == "Best Customers":
                            st.write(best_customer)
                            st.image(bestcus)
                        elif segment == "Big Spenders Customers":
                            st.write(big_spend_customer)
                            st.image(bigcus)
                        elif segment == "Lost Customers":
                            st.write(lost_customer)
                            st.image(lostcus)
                        elif segment == "Loyal Customers":
                            st.write(loyal_customer)
                            st.image(loyalcus)
                        elif segment == "Need Attention Customers":
                            st.write(need_ac_customer)
                            st.image(needcus)
                        else:
                            st.write(new_customer)
                            st.image(newcus)
                csv = df_predict.to_csv(index=False)
                st.download_button(
                    label="Download predictions as CSV file",
                    data=csv,
                    file_name="predictions.csv",
                    mime="csv",
                )

    else:

        st.write(
            """
    ##### Note:
     - ###### Please provide only the data file in CSV format.
     - ###### Please submit the data file in the following format:
    """
        )
        st.write(rfm_data.drop("CustomerID", axis=1).sample(3))
        uploaded_file = st.file_uploader("Select file data", type=["csv"])
        if uploaded_file is not None:
            st.write("##### Your data:")
            # Read file data
            data_new = pd.read_csv(uploaded_file, sep=",")
            st.write(data_new)
            df_predict = segment(data_new, rfm_data)
            df_predict = df_predict.reset_index(drop=True)
            submitted = st.button("Submit")
            if submitted:
                st.write("##### II. Result")
                segments = rfm_agg["Segment"].unique()
                for segment in segments:
                    # Filter df by segment
                    filtered_df = df_predict[df_predict["Segment"] == segment]
                    # Print df after filter
                    if not filtered_df.empty:
                        st.write(f"###### Segment: {segment}")
                        st.write(filtered_df)
                        if segment == "Best Customers":
                            st.write(best_customer)
                            st.image(bestcus)
                        elif segment == "Big Spenders Customers":
                            st.write(big_spend_customer)
                            st.image(bigcus)
                        elif segment == "Lost Customers":
                            st.write(lost_customer)
                            st.image(lostcus)
                        elif segment == "Loyal Customers":
                            st.write(loyal_customer)
                            st.image(loyalcus)
                        elif segment == "Need Attention Customers":
                            st.write(need_ac_customer)
                            st.image(needcus)
                        else:
                            st.write(new_customer)
                            st.image(newcus)
                csv = df_predict.to_csv(index=False)
                st.download_button(
                    label="Download predictions as CSV file",
                    data=csv,
                    file_name="predictions.csv",
                    mime="csv",
                )


# Create Find Infomation in Dataset table

else:
    st.subheader("📂Find Customer Information in Dataset")
    # Select data
    st.write("##### I. Select Type")
    select_find = st.radio(
        "Would you like to search by CustomerID or InvoiceNo?",
        options=("CustomerID", "InvoiceNo"),
    )
    if select_find == "CustomerID":
        st.write(
            "Please enter CustomerID similar to the pattern below or select from the suggestions in the box:"
        )
        st.write(data_final["CustomerID"].sample(3))
        customerid_lst = data_final["CustomerID"].drop_duplicates().values
        customerid = st.selectbox("Choose One Customer", customerid_lst)
        submitted = st.button("Submit")
        if submitted:
            st.write("##### II. Result")
            segment = data_final[data_final["CustomerID"] == customerid][
                "Segment"
            ].values[0]
            recency = rfm_data[rfm_data["CustomerID"] == customerid]["Recency"].values[
                0
            ]
            frequency = rfm_data[rfm_data["CustomerID"] == customerid][
                "Frequency"
            ].values[0]
            monetary = rfm_data[rfm_data["CustomerID"] == customerid][
                "Monetary"
            ].values[0]
            country = data_final[data_final["CustomerID"] == customerid][
                "Country"
            ].values[0]
            st.write("Segment:", segment)
            st.write("Recency:", recency)
            st.write("Frequency:", frequency)
            st.write("Monetary:", monetary)
            st.write("Country:", country)
            text = " ".join(
                data_final[data_final["CustomerID"] == customerid]["Description"]
            )
            wordcloud = WordCloud(
                background_color="white", width=800, height=400, max_words=50
            ).generate(text)
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.set_title("What did the Customer buy?", fontsize=25)
            ax.axis("off")
            st.pyplot(fig)
            data_customerid = {
                "CustomerID": customerid,
                "Segment": segment,
                "Recency": recency,
                "Frequency": frequency,
                "Monetary": monetary,
                "Country": country,
                "Description": ", ".join(
                    data_final[data_final["CustomerID"] == customerid]["Description"]
                ),
            }
            df_customerid = pd.DataFrame(data_customerid, index=[0])
            # st.write(df_customerid)
            csv = df_customerid.to_csv(index=False)
            st.download_button(
                label="Download Customer info as CSV file",
                data=csv,
                file_name="customer_info.csv",
                mime="csv",
            )
    else:
        st.write(
            "Please enter InvoiceNo similar to the pattern below or select from the suggestions in the box:"
        )
        st.write(data_final["InvoiceNo"].sample(3))
        invoiceno_lst = data_final["InvoiceNo"].drop_duplicates().values
        invoiceno = st.selectbox("Choose One InvoiceNo", invoiceno_lst)
        submitted = st.button("Submit")
        if submitted:
            st.write("##### II. Result")
            customerid = data_final[data_final["InvoiceNo"] == invoiceno][
                "CustomerID"
            ].values[0]
            segment = data_final[data_final["CustomerID"] == customerid][
                "Segment"
            ].values[0]
            recency = rfm_data[rfm_data["CustomerID"] == customerid]["Recency"].values[
                0
            ]
            frequency = rfm_data[rfm_data["CustomerID"] == customerid][
                "Frequency"
            ].values[0]
            monetary = rfm_data[rfm_data["CustomerID"] == customerid][
                "Monetary"
            ].values[0]
            country = data_final[data_final["CustomerID"] == customerid][
                "Country"
            ].values[0]
            st.write("Segment:", segment)
            st.write("Recency:", recency)
            st.write("Frequency:", frequency)
            st.write("Monetary:", monetary)
            st.write("Country:", country)
            text = " ".join(
                data_final[data_final["CustomerID"] == customerid]["Description"]
            )
            wordcloud = WordCloud(
                background_color="white", width=800, height=400, max_words=50
            ).generate(text)
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.set_title("What did the Customer buy?", fontsize=25)
            ax.axis("off")
            st.pyplot(fig)
            data_customerid = {
                "CustomerID": customerid,
                "Segment": segment,
                "Recency": recency,
                "Frequency": frequency,
                "Monetary": monetary,
                "Country": country,
                "Description": ", ".join(
                    data_final[data_final["CustomerID"] == customerid]["Description"]
                ),
            }
            df_customerid = pd.DataFrame(data_customerid, index=[0])
            # st.write(df_customerid)
            csv = df_customerid.to_csv(index=False)
            st.download_button(
                label="Download Customer info as CSV file",
                data=csv,
                file_name="customer_info.csv",
                mime="csv",
            )
