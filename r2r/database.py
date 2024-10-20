class DatabaseHelper:

    def ingest_files(self, filepaths: list[str]): 
            """
            Ingest files into postgres(pgvector). 
            If a document with the same title is already present in the database, nothing gets embedded.
            Invalid filepaths are ignored.

            Args:
                file_paths (list[str]): List of file paths to ingest. Should be locally available.
                
            Returns:
                None
            """
            for filepath in filepaths:
                try:
                    self.__client.ingest_files(file_paths=[filepath])
                    print(f'Ingested: {filepath} ...')
                except R2RException as r2re:
                    print(self.__parse_r2r_error(r2re))
                except Exception as e:
                    print(e)
        
    def ingest_chunks(self, chunks: list[dict], metadata: dict[str] = None):
        """
        Ingest chunks of a document into postgres(pgvector). 
        If a document with the same title is already present in the database, nothing gets embedded.

        Args:
            chunks (list[str]): List of document pieces to ingest.
            metadata Optional(dict[str]): Dictionary of metadata to associate with the document.

        Returns:
            list[dict]: Ingestion results containing message, task_id and a document_id.
        """    
        try:
            return self.__client.ingest_chunks(chunks=chunks, metadata=metadata)['results']
        except R2RException as r2re:
            print(self.__parse_r2r_error(r2re))
        except Exception as e:
            print(e)
    
    def update_files(self, filepaths, document_ids):    
        """
        Update files in the database. If a document with the same title is already present in the database, it gets updated.

        Args:
            file_paths (list[str]): List of file paths to update.
            document_ids (list[str]): List of document IDs to associate with the updated files.

        Raises:
            ValueError: If the lengths of filepaths and document_ids are not equal.    

        Returns:
            int: Files updated.
        """
        if len(filepaths) != len(document_ids):
            raise ValueError("Filepaths and document_ids must have the same length.")
        
        files_updated = 0
        for filepath, document_id in zip(filepaths, document_ids):
            try:
                self.__client.update_files(filepaths, document_ids)    
                files_updated += 1
            except R2RException as r2re:
                print(self.__parse_r2r_error(r2re))
            except Exception as e:
                print(e)
        return files_updated
    
    def documents_overview(self, documents_ids=None, offset=0, limit=100):
        """
        Get an overview of documents in the database.

        Args:
            documents_ids (list[str], optional): List of document IDs to get overview of. If not provided, overview of all documents is returned.

        Returns:
            list[dict]: List of dictionaries containing document_id, version, title, etc.
        """
        return self.__client.documents_overview(documents_ids, offset, limit)["results"]
        
    def document_chunks(self, document_id, offset=0, limit=100, include_vectors=False):  
        """
        Get chunks of a document.

        Args:
            document_id (str): ID of the document to get chunks of.
            offset Optional(int): Offset of the first chunk to return. Defaults to 0.
            limit Optional(int): Maximum number of chunks to return. Defaults to 100.
            include_vectors Optional(bool): Whether to include embeddings in the returned chunks. Defaults to False.

        Returns:
            list[dict]: List of dictionaries containing chunk_id, text, embeddings, etc.
        """
        try:
            return self.__client.document_chunks(document_id)['results']
        except R2RException as r2re:
            return self.__parse_r2r_error(r2re)
        except Exception as e:
            return e
            
    def delete(self, filters: list[dict]):
        """
        Delete documents from the database based on filters.

        Args:
            filters (list[dict]): List of dictionaries containing filter criteria.
            Example: "document_id": {"$eq": "9fbe403b-c11c-5aae-8ade-ef22980c3ad1"}
        Returns:
            None
        """    
        files_deleted = 0
        for filter in filters:
            try:
                self.__client.delete(filter)
                files_deleted += 1
            except R2RException as r2re:
                print(self.__parse_r2r_error(r2re))
            except Exception as e:
                print(e)
        return files_deleted
    
    def clean_db(self):
        try:
            docs_metadata = self.documents_overview()
            filters = [{"document_id": {"$eq": doc_metadata["document_id"]}} for doc_metadata in docs_metadata]
            self.delete(filters)
        except R2RException as r2re:
            print(self.__parse_r2r_error(r2re))
        except Exception as e:
            print(e)