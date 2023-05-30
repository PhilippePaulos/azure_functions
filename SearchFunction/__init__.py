import io
import json
import logging
import os
import pickle

import azure.functions as func
import numpy as np
import pandas as pd
import redis
from azure.storage.blob import BlobServiceClient
from haversine import haversine_vector, Unit
from scipy.spatial import KDTree


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")

    latitude = req.params.get("latitude")
    longitude = req.params.get("longitude")
    radius = req.params.get("radius")

    if not all([latitude, longitude, radius]):
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            latitude = req_body.get("latitude")
            longitude = req_body.get("longitude")
            radius = req_body.get("radius")

    if all([latitude, longitude, radius]):
        try:
            latitude = float(latitude)
            longitude = float(longitude)
            radius = int(radius)

            # Connect to Redis
            r = redis.Redis(
                host=os.getenv('AZURE_REDIS_HOST'),
                port=os.getenv('AZURE_REDIS_PORT'),
                password=os.getenv('AZURE_REDIS_KEY'),
                ssl=True,
                ssl_cert_reqs=None
            )

            result = r.ping()
            logging.info("Ping returned : " + str(result))

            data = r.get('data')
            kdtree = r.get('kdtree')

            if data is None or kdtree is None:
                logging.info("Loading data from blob storage")
                # Get Azure Blob Storage connection string
                blob_connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")

                # Get Azure Blob Storage
                container = os.getenv("AZURE_BLOB_CONTAINER")
                blob_data = os.getenv("AZURE_BLOB_DATA")
                blob_kdtree = os.getenv("AZURE_BLOB_KDTREE")

                # Download data and KDTree from Blob Storage
                blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
                data_blob_client = blob_service_client.get_blob_client(container, blob_data)
                kdtree_blob_client = blob_service_client.get_blob_client(container, blob_kdtree)

                data_blob_bytes = data_blob_client.download_blob().readall()
                kdtree_blob_bytes = kdtree_blob_client.download_blob().readall()

                # Load the downloaded data into a DataFrame and the KDTree
                data_df = pd.read_json(io.BytesIO(data_blob_bytes))
                kdtree = pickle.loads(kdtree_blob_bytes)

                # Cache the data and kdtree in Redis
                r.set('data', data_df.to_json())
                r.set('kdtree', pickle.dumps(kdtree))
            else:
                # Data is in cache, load it
                logging.debug("Loading data from redis")
                data_df = pd.read_json(data.decode("utf-8"))
                kdtree = pickle.loads(kdtree)

            # Create an instance of the SearchProcess class and use it to compute distances
            nearby_restaurants_df = compute_distances(data_df, kdtree, latitude, longitude, radius)

            return func.HttpResponse(f"{json.dumps(nearby_restaurants_df.to_dict(), ensure_ascii=False)}")
        except Exception as e:
            return func.HttpResponse(
                f"{e}",
                status_code=400
            )
    else:
        return func.HttpResponse(
            "Missing parameters. Please provide latitude, longitude, and radius.",
            status_code=400
        )


def compute_distances(df: pd.DataFrame, kdtree: KDTree, latitude: float, longitude: float, radius: int) -> pd.DataFrame:
    point = np.array([latitude, longitude])

    # Convert the radius from meters to degrees and add buffer space
    radius_in_degrees = radius / 110574 + 0.01

    # Use KDTree to find points within radius
    indices = kdtree.query_ball_point(point, radius_in_degrees)
    restaurants_near_centroid = df.iloc[indices].copy()

    coordinates = restaurants_near_centroid[["latitude", "longitude"]].values
    distances = haversine_vector(np.array([latitude, longitude]), coordinates,
                                 unit=Unit.METERS,
                                 comb=True)

    # Round distances to 2 decimal places
    distances = np.round(distances, 2)

    # Filter restaurants based on actual search radius
    mask = distances <= radius
    return restaurants_near_centroid[mask].assign(distance=distances[mask])
