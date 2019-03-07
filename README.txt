Procedure to run the model
======================

1. Add __init__.py to create module. The file could not be uploaded here as its an empty file.
2. Run product_classifier -
	This shows the performance of various models
	and persists Linear SVC model
3. Run product_classifier_api - 
	This will bring up the server and run from port 10001
4. Use make_api_request -
	This will pass in various production descriptions, for which the product category is predicted and returned in the response.
