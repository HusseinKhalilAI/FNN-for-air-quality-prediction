from shiny import App, ui, render, reactive
import matplotlib.pyplot as plt
import pandas as pd
from model import PM_Model
import torch
import pickle as pkl

pm_pred_app = ui.page_fluid(
    ui.layout_sidebar(
        ui.sidebar(

            ui.input_file("upload", "Upload Data File"),
            ui.output_ui("upload_inst"),
            ui.output_ui("pollutants_selectize"),
            ui.output_ui("time_window"),
            ui.output_ui("pred_checkmark"),
            ui.output_ui("scaler_weights_upload"),
        ),
    
        ui.output_plot("pm_plot")  

    )
)
    

def server(input, output, session):

    @output
    @render.ui
    def upload_inst():
        if input.upload() is None:
            return ui.p("Upload a CSV file to continue")
        return None
    
    @output
    @render.ui
    def pollutants_selectize():

        file = input.upload()   
        if file is None:
            return None
        
        df=pd.read_csv(file[0]["datapath"])
        
        
        return ui.input_selectize(
            "pollutants",
            "Select Pollutants",
            choices=["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"],
            multiple=True,
            selected=["PM2.5"]
        )
    
    @output
    @render.ui
    def time_window():
        file = input.upload()   
        if file is None:
            return None
        
        df=pd.read_csv(file[0]["datapath"])
        
        return ui.input_slider("smoothing", "Smoothing Window (days)", 1, 30, 1)
    @output
    @render.ui
    def pred_checkmark():
        file = input.upload()   
        if file is None:
            return None
        return ui.input_checkbox("pm25_pred", "Show PM2.5 Prediction")
    

    @output
    @render.ui
    def scaler_weights_upload():
        if input.pm25_pred() is True:
            return ui.TagList(
                ui.input_file("scalers", "Upload scaler (.pkl)"),
                ui.input_file("weights", "Upload weights (.pt)")
            )
        return None
           

    @output
    @render.plot
    def pm_plot():

        file = input.upload()
        if file is None:
            return None
    
        path = file[0]["datapath"]
        df = pd.read_csv(path)

        window_days = input.smoothing()

        selected_pollutants = input.pollutants() 
        cols = ["year", "month", "day"]
        df["date"] = pd.to_datetime(df[cols])

   

          
        df = df.set_index("date").sort_index()
        date_time = df[list(selected_pollutants)].resample("D").mean()
        adj = date_time.rolling(window=window_days, min_periods=1).mean().dropna()

        fig, ax = plt.subplots()

        for col in selected_pollutants:
            ax.plot(adj.index, adj[col], label=col)

        ax.set_title("Pollution Levels")
        ax.set_xlabel("DateTime")
        ax.set_ylabel("Value")
        ax.legend()

        if input.pm25_pred() is True:
            weights_file = input.weights()
            scalers_file = input.scalers()

            
            if weights_file is None or scalers_file is None:
                return fig  

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            df = df.select_dtypes(include="number").dropna()

            trained_features = df.drop(columns=["PM2.5", "No",'Unnamed: 0']).reset_index(drop=True)

            model = PM_Model(
                input_size=trained_features.shape[1],
                hidden_size=64,
                hidden_layers=3,
                drop_out=True,
                drop_value=0.2
            )

            model.load_state_dict(torch.load(weights_file[0]["datapath"], map_location=device))
            model.eval()

    
            with open(scalers_file[0]["datapath"], "rb") as f:
                scalers = pkl.load(f)

            x= df.drop(columns=["PM2.5", "No",'Unnamed: 0']).reset_index(drop=True)
            X = scalers.transform(x)
            X_tensor = torch.tensor(X, dtype=torch.float32)

            with torch.no_grad():
                y = model(X_tensor).numpy().flatten()

            df["PM2.5_predictions"] = y
            pred = df["PM2.5_predictions"]
            adj_pre = pred.rolling(window=window_days, min_periods=1).mean().dropna()

            ax.plot(df.index, adj_pre, label="PM2.5 predictions",  linestyle="--", alpha=0.3)
            ax.legend()
            




app = App(pm_pred_app, server)