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

# Match gift using text embeddings for similarity
def match_gift(state: WishListWizardsState):
    print("Matching generated gifts with warehouse items using embeddings...")
    _, warehouse_data = load_data()
    generated_gifts = state['generated_gift']
    matched_gifts = {}

    # Flatten warehouse items
    warehouse_items = warehouse_data['item'].tolist()

    # Check if warehouse_items list is empty
    if not warehouse_items:
        print("Error: Warehouse items list is empty!")
        return state

    # Compute embeddings for warehouse items
    print("Generating embeddings for warehouse items...")
    warehouse_embeddings = embed_model(warehouse_items)

    for child, gift_options in generated_gifts.items():
        best_match = "No Match Found"
        max_similarity = -1

        # Generate embeddings for the child's gift preferences
        print(f"Processing gift preferences for: {child}")
        gift_embeddings = embed_model(gift_options)

        # Compute similarity with warehouse items
        similarities = cosine_similarity(gift_embeddings, warehouse_embeddings)
        print(f"Similarity matrix for {child}:\n{similarities}")

        # Find the best match
        for i, similarity_row in enumerate(similarities):
            max_index = np.argmax(similarity_row)
            similarity_score = similarity_row[max_index]

            # Check if the max index is valid
            if 0 <= max_index < len(warehouse_items):
                if similarity_score > max_similarity and similarity_score > 0.5:  # Threshold
                    max_similarity = similarity_score
                    best_match = warehouse_items[max_index]
            else:
                print(f"Warning: max_index {max_index} is out of range for warehouse_items.")

        # Store the best match for the child
        matched_gifts[child] = best_match
        print(f"Best match for {child}: {best_match} (Similarity: {max_similarity:.3f})")

    state['matched_gifts'] = matched_gifts
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


import pandas as pd
from langgraph.graph import START, END, StateGraph
from typing import TypedDict, List, Dict

# Define State
class WishListWizardsState(TypedDict):
    generated_gift: Dict[str, List[str]]  # Stores generated gifts per child
    matched_gifts: Dict[str, str]        # Matched gifts per child

# Load data from the CSV files
def load_data():
    children_data = pd.read_csv('children_list.csv')  # Replace with file path
    warehouse_data = pd.read_csv('warehouse.csv')     # Replace with file path
    return children_data, warehouse_data

# Generate gift based on child likes
def generate_gift(state: WishListWizardsState):
    print("Generating gifts based on children's preferences...")
    children_data, _ = load_data()
    generated_gifts = {}

    for _, row in children_data.iterrows():
        likes = row['likes'].split(',')  # Get the child's likes
        generated_gifts[row['full_name']] = likes  # Assume likes as possible gifts

    state['generated_gift'] = generated_gifts
    return state

# Match gift based on warehouse availability
def match_gift(state: WishListWizardsState):
    print("Matching generated gifts with warehouse items...")
    _, warehouse_data = load_data()
    generated_gifts = state['generated_gift']
    matched_gifts = {}

    # Flatten warehouse items for easy searching
    available_items = warehouse_data['item'].tolist()

    for child, gift_options in generated_gifts.items():
        for gift in gift_options:
            if gift in available_items:
                matched_gifts[child] = gift
                break
        else:
            matched_gifts[child] = "No Match Found"

    state['matched_gifts'] = matched_gifts
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
    print("Final Matched Gifts:")
    for child, gift in final_state['matched_gifts'].items():
        print(f"{child}: {gift}")

place,item,quantity
Hamburg Warehouse,basketball,40
Hamburg Warehouse,swimming,60
Hamburg Warehouse,Scooter,10
Hamburg Warehouse,Trampoline,4
Hamburg Warehouse,Dollhouse,20
Hamburg Warehouse,Train Set,40
Hamburg Warehouse,Action Figures,20
Hamburg Warehouse,Walkie Talkies,4
Hamburg Warehouse,Toy Kitchen Set,2
Hamburg Warehouse,Magic Set,20
Hamburg Warehouse,Science Kit,6
Hamburg Warehouse,Building Blocks,20


full_name,age,city,country,last_year_gift,likes
David Williams,4,Bristol,England,Football,"basketball,swimming"
Maria Garcia,7,Madrid,Spain,Dancing,"Singing,Dolls"
John Smith,6,New York,USA,Video Games,Lego Set
Anna Müller,5,Berlin,Germany,Drawing,"Painting,Art Supplies"
Sophia Lee,8,Seoul,South Korea,Reading,"Swimming,Books"
Benjamin Rodriguez,9,Mexico City,Mexico,Soccer,Soccer Ball
Olivia Nguyen,3,Hanoi,Vietnam,Building Blocks,Blocks
Noah Brown,10,London,England,Lego,"Basketball,Lego City"
Emma Kim,7,Busan,South Korea,Piano,"Dancing,Musical Instrument"
Liam Wilson,6,Manchester,England,Cars,Toy Car
Ava Martinez,5,Barcelona,Spain,Animals,Stuffed Animal
Isabella Chen,8,Taipei,Taiwan,Drawing,"Reading,Art Set"
