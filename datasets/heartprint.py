
import os
import glob
from typing import Dict, List, Optional


class HeartPrintDataset:
    """
    Class to traverse and organize the HeartPrint dataset directory into a nested dictionary format:
    {
        "Session-1": {
            "001": [file1_path, file2_path, ...],
            ...
        },
        ...
    }
    """

    def __init__(self, base_path: str, sessions: Optional[List[str]] = None):
        """
        :param base_path: Path to the Heartprint/Heartprint/ folder
        :param sessions: List of session names to load (optional)
        """
        self.base_path = base_path
        self.sessions = sessions if sessions else ["Session-1", "Session-2", "Session-3L", "Session-3R"]

    def get_all_recordings(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Load all ECG file paths grouped by session and subject.

        :return: Dictionary[session][subject_id] -> list of file paths
        """
        dataset = {}

        for session in ["Session-1", "Session-2", "Session-3L", "Session-3R"]:
            session_path = os.path.join('C:\\Users\\milad\\Desktop\\ecg-biometrics\\datasets\\Heartprint', session)
            if not os.path.isdir(session_path):
                print(f"[Warning] Session folder not found: {session_path}")
                continue

            dataset[session] = {}
            subject_folders = sorted(os.listdir(session_path))

            for subject_id in subject_folders:
                subject_path = os.path.join(session_path, subject_id)
                if not os.path.isdir(subject_path):
                    continue

                ecg_files = glob.glob(os.path.join(subject_path, '*'))
                dataset[session][subject_id] = ecg_files

        return dataset


        
        
        
        
        
        
        
