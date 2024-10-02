import gradio as gr
import requests

# Define the API endpoint
API_URL = "http://fastapi_api:8000/predict"

# Define the function to call the API
def predict(
    Company, TypeName, Inches, Ram, OS, Weight, Screen, ScreenW, ScreenH, 
    Touchscreen, IPSpanel, RetinaDisplay, 
    CPU_company, CPU_freq, PrimaryStorage, 
    SecondaryStorage, PrimaryStorageType, 
    SecondaryStorageType, GPU_company):
    
    # Create a payload with the input data
    payload = {
        "Company": Company,
        "TypeName": TypeName,
        "Inches": Inches,
        "Ram": Ram,
        "OS": OS,
        "Weight": Weight,
        "Screen": Screen,
        "ScreenW": ScreenW,
        "ScreenH": ScreenH,
        "Touchscreen": Touchscreen,
        "IPSpanel": IPSpanel,
        "RetinaDisplay": RetinaDisplay,
        "CPU_company": CPU_company,
        "CPU_freq": CPU_freq,
        "PrimaryStorage": PrimaryStorage,
        "SecondaryStorage": SecondaryStorage,
        "PrimaryStorageType": PrimaryStorageType,
        "SecondaryStorageType": SecondaryStorageType,
        "GPU_company": GPU_company,
    }
    
    # Send a POST request to the FastAPI endpoint
    response = requests.post(API_URL, json=payload)
    
    if response.status_code == 200:
        return response.json()["prediction"]
    else:
        return f"Error: {response.text}"

# Create Gradio interface
inputs = [
    gr.Dropdown(['Apple','HP','Acer','Asus','Dell','Lenovo','Chuwi','MSI','Microsoft','Toshiba',
                 'Huawei', 'Xiaomi', 'Vero','Razer','Mediacom','Samsung','Google','Fujitsu','LG'],label="Company"),
    gr.Dropdown(['Ultrabook','Notebook','Netbook','Gaming','2 in 1 Convertible','Workstation'],label="TypeName"),
    gr.Number(label="Inches"),
    gr.Number(label="Ram"),
    gr.Dropdown(['macOS','No OS','Windows 10','Mac OS X','Linux','Android','Windows 10 S','Chrome OS','Windows 7'],label="OS"),
    gr.Number(label="Weight"),
    gr.Dropdown(['Standard','Full HD','Quad HD+','4K Ultra HD'],label="Screen"),
    gr.Number(label="ScreenW"),
    gr.Number(label="ScreenH"),
    gr.Dropdown(['Yes','No'],label="Touchscreen (Yes/No)"),
    gr.Dropdown(['Yes','No'],label="IPSpanel (Yes/No)"),
    gr.Dropdown(['Yes','No'],label="RetinaDisplay (Yes/No)"),
    gr.Dropdown(['Intel', 'AMD', 'Samsung'],label="CPU_company"),
    gr.Number(label="CPU_freq"),
    gr.Number(label="PrimaryStorage"),
    gr.Number(label="SecondaryStorage"),
    gr.Dropdown(['SSD', 'Flash Storage', 'HDD', 'Hybrid'],label="PrimaryStorageType"),
    gr.Dropdown(['No' ,'HDD', 'SSD', 'Hybrid'],label="SecondaryStorageType"),
    gr.Dropdown(['Intel', 'AMD', 'Nvidia', 'ARM'],label="GPU_company"),
]

output = gr.Label(label="Predicted Price (Euros)")
app = gr.Interface(fn=predict, inputs=inputs, outputs=output, title="Laptop Price Predictor", description="Enter the details of the laptop to get the predicted price in Euros.")
app.launch(server_name="0.0.0.0", server_port=7860)

