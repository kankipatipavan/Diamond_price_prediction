import streamlit as st
import pandas as pd
import numpy as np

def Login():
    import yaml
    import streamlit_authenticator as stauth
    from yaml.loader import SafeLoader

    with open('config.yml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )
    st.write("# Diamond Price Predictor 💎")
    authenticator.login('Login', 'main')
    if st.session_state["authentication_status"]:
        authenticator.logout('Logout', 'main', key='unique_key')
        st.write(f'Welcome *{st.session_state["name"]}*')
        
    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] is None:
        st.warning('Please enter your username and password')
    

def Home():
    import streamlit as st

    st.write("# Diamond Grading System 💍")
    st.sidebar.success("Pleae select for more content.")

    st.markdown("""CARAT – a Diamond’s Weight
    Diamonds have a unique scale for weight in “carats”. One carat is 0.2 grams, or 200 milligrams.""")
    st.image("dweight.png", width= 800)
    st.markdown("""
        COLOR – a Diamond’s Tint
        GIA grades diamond colors from D (totally colorless) to Z (yellow). The difference between each grade is small and usually only obvious to a trained expert.
        The most common color range for diamonds used in engagement rings is G-J (near colorless).
        There are some stones that have chemical impurities that tint the diamond a different color such as blue, green or pink. These are called fancy diamonds and are rare and costly.""")
    st.image("dtint.png", width= 800)
    st.markdown("""
        CLARITY – a Diamond’s Purity
        Clarity judges how free the diamond is from imperfections. Clarity grades range from internally flawless (IF) to fairly included (I3).
        To most people, clarity is the least important of the four Cs. This is because the difference in some grades is very difficult to see except under magnification, and the difference between an SI1 and a VVS2 diamond may not be noticed except by a professional gemologist.""")
    
    st.image("dpurity.png", width= 600)
    st.markdown("""
        CUT – a Diamond’s Brilliance
        The cut is not the shape of the diamond, but refers to the cutting of the diamond into facets and the way in which it reflects light internally.
        The cut is the primary factor in the brilliance or fire of a diamond, where light seems to come from the very heart of the diamond itself.""")
    st.image("dbrilliance.png", width= 500)
    st.markdown("""
        Dimensions
        The higher the depth percentage, the deeper the stone appears. For an ideal cut round diamond, the best depth percentage is 58%-62.9%.

        The higher the table percentage, the bigger the table looks. For an ideal cut round diamond, the best table percentage is 53%-57%.

        Reference: Rice Village Diamonds, Diamond Education""")
    st.image("dimensions.png", width= 600)

def Case_study():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.title("Casestudy analysis on Diamonds dataset")
    st.image("diamond.png", width= 600)
    df = pd.read_csv(r'diamonds.csv')
    data = df.drop(df.columns[0],axis=1)
    numdata = data.select_dtypes(include='number')
    
    st.markdown(""" Shape of the dataset""")
    st.write(data.shape, 'rows, columns')

    st.header('Tabular data of the dataset')
    if st.checkbox("tabular data"):
        st.table(data.head(10))
    
    st.header("Data types present in the data")
    if st.checkbox("Check datatypes"):
        st.table(data.dtypes)
    
    st.header("Missing values present in the data")
    if st.checkbox("Check Missing data"):
        st.table(data.isna().sum(axis=0))

    st.header("Statistical summary of the Diamond data")
    if st.checkbox("Statistics"):
        st.table(data.describe())

    if st.header("Corelation analysis"):
        fig,ax=plt.subplots(figsize=(9,4))
        sns.heatmap(numdata.corr(),annot=True, cmap='coolwarm')
        st.pyplot(fig)
        st.markdown("""
Interpretations of correlational relationships

1. There is an increase in carat is associated with raise in price.

2. Compared to Poor cut, the raise in prices is gradually high for Good cut, Very Good cut, Premium cut, Ideal cut.""")
    
    st.title("Graphs")
    graph = st.selectbox("Different types of graphs", ["Scatter plot","Histogram", "Bar graph" ])
    if graph =="Scatter plot":
        value = st.slider('Filter data using carat', 0,6)
        data = data.loc[data["carat"]>= value]
        fig, ax=plt.subplots(figsize=(10,5))
        sns.scatterplot(data=data, x= 'carat', y = 'price', hue = 'cut')
        st.pyplot(fig)
    if graph =="Bar graph":
        fig, ax=plt.subplots(figsize =(5,2))
        sns.barplot(x="cut",y=data.cut.index,data=data)
        st.pyplot(fig)
    if graph =="Histogram":
        fig,ax=plt.subplots(figsize =(5,3))
        sns.distplot(data.price,kde=True)
        st.pyplot(fig)

    st.title("Box Plot")
    fig, axs = plt.subplots(3, 1, layout='tight', figsize=(8,7))
    sns.boxplot(data, x='cut', y='carat', showfliers=False, ax=axs[0])
    sns.boxplot(data, x='clarity', y='carat', showfliers=False, ax=axs[1])
    sns.boxplot(data, x='color', y='carat', showfliers=False, ax=axs[2])
    st.pyplot(fig)
    st.markdown("""
        #### Carat tends to be lower when other qualities are better, possibly due to the increased difficulty to discover larger gems with outstanding qualities.""")

    st.markdown(
        """
        #### Motivated by the above EDA, I would include the '4Cs' first.

        #### I use a log-log transformation for price and carat that renders a beautiful linear relationship.
    """
    )


def Prediction():

    st.title("Prediction price of a Diamond ")
    df = pd.read_csv(r'diamonds.csv')
    data = df.drop(df.columns[0],axis=1)
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    X = np.array(data['carat']).reshape(-1,1)
    y = np.array(data['price']).reshape(-1,1)
    lr.fit(X,y)
    value=st.number_input("carat",0.20,5.01,step=0.15)
    value=np.array(value).reshape(1,-1)
    prediction=lr.predict(value)[0]
    if st.button("Predict($)"):
        st.write(f"{prediction}")
    st.markdown(
        """
        ### Prediction performance.

        The predictions are fairly precise for less expensive diamonds below $10,000. For more expensive ones, it should only be taken as a rough hint.
    """)


page_names_to_funcs = {
    "Login": Login,
    "Home": Home,
    "Case study": Case_study,
    "Prediction": Prediction,
}
page = st.sidebar.selectbox("Navigator", page_names_to_funcs.keys())
page_names_to_funcs[page]()