import weaviate

# connect to local Weaviate
client = weaviate.connect_to_local()

# check connection
print("Connected:", client.is_ready())
