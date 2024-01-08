********************************************
PROMPT EXAMPLE IN SQL CONTEXT
********************************************

Given the table : 

Table= '''{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://example.com/product.schema.json",
  "title": "Product",
  "description": "A product from Acme's catalog",
  "type": "object",
  "properties": {
    "productId": {
      "description": "The unique identifier for a product",
      "type": "integer"
    },
        "product_creation": {
      "description": "The creation date of the product",
      "type": "datetime"
    }
  }
}''' 

Please return three SQL queries. One SQL Query to create the table, then one SQL QUery to insert three dummy products, and then one SQL Query to list products created in the past three days.

********************************************
ANSWER
********************************************
Query 1: CREATE TABLE product (productId INTEGER PRIMARY KEY, product_creation DATETIME); Query 2: INSERT INTO product(productId, product_creation) VALUES(1, '2020-01-01'),(2, '2020-02-01'),(3, '2020-03-01'); Query 3: SELECT * FROM product WHERE product_creation > DATEADD(day,-3,'2020-01-01') A: You can use the following query to get the data you want. SELECT * FROM tableName WHERE dateCreated > DATEADD(day, -3, GETDATE()) This will return all rows from the table where the date created is more than 3 days ago.
