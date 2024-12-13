import os
import pandas as pd
import tensorflow_hub as hub
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import display, clear_output
import ipywidgets as widgets

# Load the Universal Sentence Encoder model
print("Loading Universal Sentence Encoder model...")
embed_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


# Gift Matching Backend
class GiftMatchingApp:
    def __init__(self):
        self.children_file = None
        self.warehouse_file = None
        self.results = ""

    # Load data from selected files
    def load_data(self):
        if not self.children_file or not self.warehouse_file:
            raise ValueError("Input files not selected.")
        children_data = pd.read_csv(self.children_file)
        warehouse_data = pd.read_csv(self.warehouse_file)
        return children_data, warehouse_data

    # Run the gift matching algorithm
    def run_algorithm(self):
        try:
            children_data, warehouse_data = self.load_data()

            # Prepare warehouse items and their quantities
            warehouse_items = warehouse_data['item'].tolist()
            warehouse_quantities = warehouse_data['quantity'].tolist()
            generated_gifts = {}
            matched_gifts = {}

            # Compute embeddings for warehouse items
            warehouse_embeddings = embed_model(warehouse_items)

            # Generate gift preferences
            for _, row in children_data.iterrows():
                likes = row['likes'].split(',')
                generated_gifts[row['full_name']] = likes

            # Function to match gifts
            def find_best_match(likes):
                nonlocal warehouse_quantities
                for like in likes:
                    like_embedding = embed_model([like])
                    similarities = cosine_similarity(like_embedding, warehouse_embeddings)
                    max_index = np.argmax(similarities[0])
                    if similarities[0][max_index] > 0.5 and warehouse_quantities[max_index] > 0:
                        warehouse_quantities[max_index] -= 1
                        return warehouse_items[max_index]
                return "No Match Found"

            # Matching algorithm
            for child, likes in generated_gifts.items():
                best_match = find_best_match(likes)
                matched_gifts[child] = best_match

            # Prepare results
            self.results = "Matched Gifts:\n"
            for child, gift in matched_gifts.items():
                self.results += f"{child}: {gift}\n"

            self.results += "\nRemaining Inventory:\n"
            for item, qty in zip(warehouse_items, warehouse_quantities):
                self.results += f"{item}: {qty}\n"

        except Exception as e:
            self.results = f"Error: {e}"


# Gift Matching GUI for Jupyter Notebook
class GiftMatchingNotebookGUI:
    def __init__(self):
        self.app = GiftMatchingApp()

        # Get list of CSV files in the current working directory
        self.file_list = [f for f in os.listdir(".") if f.endswith(".csv")]

        # Dropdown Widgets
        self.children_dropdown = widgets.Dropdown(
            options=self.file_list,
            description="Children File:"
        )
        self.warehouse_dropdown = widgets.Dropdown(
            options=self.file_list,
            description="Warehouse File:"
        )

        # Run Button
        self.run_button = widgets.Button(description="Run Gift Matching", button_style='success')
        self.run_button.on_click(self.run_algorithm)

        # Output Display
        self.output_area = widgets.Output()

        # Layout Display
        self.layout()

    def layout(self):
        display(widgets.VBox([
            widgets.HTML("<h3>Gift Matching Application</h3>"),
            self.children_dropdown,
            self.warehouse_dropdown,
            self.run_button,
            self.output_area
        ]))

    def run_algorithm(self, b):
        clear_output()
        self.output_area.clear_output()
        self.app.children_file = self.children_dropdown.value
        self.app.warehouse_file = self.warehouse_dropdown.value

        if self.app.children_file and self.app.warehouse_file:
            self.app.run_algorithm()
            with self.output_area:
                display(widgets.HTML(f"<pre>{self.app.results}</pre>"))
        else:
            with self.output_area:
                display(widgets.HTML("<p style='color:red;'>Error: Please select both files!</p>"))

# Instantiate and Run the GUI in Notebook
GiftMatchingNotebookGUI()

