import os

class IOHelper:

    def iterate_over_files(self, folder_path):
        """
        Iterate over all files in a given folder and its subfolders.

        Args:
            folder_path (str): Path to the folder to iterate over.

        Yields:
            str: Path to each file.

        Returns:
            list[str]: List of all file names in the folder and its subfolders.
        """
        filenames = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                yield file_path
                filenames.append(file)
        return filenames