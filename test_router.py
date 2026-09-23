from app import route
if __name__ == "__main__":
    for f in ["hi", "What is diabetes?", "latest studies on asthma biologics",
              "How do I fix my car?"]:
        print(route(f), "|", f)
