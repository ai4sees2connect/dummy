import pandas as pd
import tensorflow_hub as hub
import numpy as np
from langgraph.graph import START, END, StateGraph
from typing import TypedDict, List, Dict
from sklearn.metrics.pairwise import cosine_similarity

# Define State
class WishListWizardsState(TypedDict):
    generated_gift: Dict[str, List[str]]  # Stores generated gifts per child
    matched_gifts: Dict[str, str]        # Matched gifts per child

# Load the Universal Sentence Encoder model from TensorFlow Hub
print("Loading Universal Sentence Encoder model...")
embed_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Load data from the CSV files
def load_data():
    children_data = pd.read_csv('children_list.csv')
    warehouse_data = pd.read_csv('warehouse.csv')
    return children_data, warehouse_data

# Generate gifts based on children's preferences
def generate_gift(state: WishListWizardsState):
    print("Generating gifts based on children's preferences...")
    children_data, _ = load_data()
    generated_gifts = {}

    for _, row in children_data.iterrows():
        likes = row['likes'].split(',')
        generated_gifts[row['full_name']] = likes  # Possible gift preferences

    state['generated_gift'] = generated_gifts
    return state

# Match gift using text embeddings and item quantities
def match_gift(state: WishListWizardsState):
    print("Matching generated gifts with warehouse items using embeddings and quantities...")
    _, warehouse_data = load_data()
    generated_gifts = state['generated_gift']
    matched_gifts = {}

    # Prepare warehouse items and their quantities
    warehouse_items = warehouse_data['item'].tolist()
    warehouse_quantities = warehouse_data['quantity'].tolist()

    # Compute embeddings for warehouse items
    print("Generating embeddings for warehouse items...")
    warehouse_embeddings = embed_model(warehouse_items)

    # Function to find the best available match
    def find_best_match(likes):
        nonlocal warehouse_quantities
        best_match = "No Match Found"
        for like in likes:
            # Generate embedding for the current like
            like_embedding = embed_model([like])
            similarities = cosine_similarity(like_embedding, warehouse_embeddings)
            max_index = np.argmax(similarities[0])
            similarity_score = similarities[0][max_index]

            # Check if the item is available
            if similarity_score > 0.5 and warehouse_quantities[max_index] > 0:
                best_match = warehouse_items[max_index]
                warehouse_quantities[max_index] -= 1  # Deduct the quantity
                print(f"Assigned '{best_match}' for '{like}' (Remaining Quantity: {warehouse_quantities[max_index]})")
                return best_match
        return best_match

    # Iterate through children and assign gifts
    for child, likes in generated_gifts.items():
        print(f"\nProcessing preferences for: {child}")
        best_match = find_best_match(likes)
        matched_gifts[child] = best_match
        print(f"Best match for {child}: {best_match}")

    state['matched_gifts'] = matched_gifts

    # Print remaining warehouse inventory
    print("\nRemaining Warehouse Inventory:")
    for item, quantity in zip(warehouse_items, warehouse_quantities):
        print(f"{item}: {quantity}")

    return state

# Main StateGraph workflow
wish_list_wizards_agent_builder = StateGraph(WishListWizardsState)
wish_list_wizards_agent_builder.add_node("generate_gift", generate_gift)
wish_list_wizards_agent_builder.add_node("match_gift", match_gift)

# Define the workflow edges
wish_list_wizards_agent_builder.add_edge(START, "generate_gift")
wish_list_wizards_agent_builder.add_edge("generate_gift", "match_gift")
wish_list_wizards_agent_builder.add_edge("match_gift", END)

# Compile the agent
wish_list_wizards_agent = wish_list_wizards_agent_builder.compile()

# Initialize and execute the graph
if __name__ == "__main__":
    initial_state = WishListWizardsState(generated_gift={}, matched_gifts={})
    final_state = wish_list_wizards_agent.invoke(initial_state)

    # Print results
    print("\nFinal Matched Gifts:")
    for child, gift in final_state['matched_gifts'].items():
        print(f"{child}: {gift}")
