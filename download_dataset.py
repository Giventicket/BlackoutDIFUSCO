import dropbox
 
dbx = dropbox.Dropbox(
    oauth2_access_token="sl.u.AFiOLxdrV7bP4vrUBL5CxTLw7CNESlRjRqiMjtpKCOKXrzNZ5VHtekom2PMYbsAAMcJRN_uupgjjFtf5TEPMT0VL_h9CbbRSNHnvWWLSe7Z2HUMP-rqhSreREocnOuHK8wNbj8qBz08UacY4F2LacT2xJy6G4cW0QQnpRatHmdolRdeO6IimAGdk5o3nrtqFDNakurbtrJgUgPtPsbtaW9gBc7HGA2PqgE1PMYyiWHq4e8DsrV4x2vprpefOA5yqOUZBGXJ2ui0S7dQ5avpB7tkp5hWWNq0JC4jjtrBM9CjnqxTUh0owlcanNtUQ50ltO_iArxizBxhfTWn7mwPwnLVHCL7NfunHVVqVeY-gjVC4tnP2uh2o67vBIhNdBGxfOnI3LxD8Ta7sGyHJVk5hyG7-UxGFSJIdYyQuV4YcwFTfgipvpa0Mh1FJ9c__9KPyCArNEqN06pwzMvFCANiIkxMLSRelbgVFjnLzWD6w5s1Cx4LgxZ0U2QIqlydKUavd68PlUGl4lzSk4d5Dm2l9NBHmRBEgBZfDz-VXSxsRLPZCVoJD44PRAnH9P86j14K3Ehp8IOY8bBrRXpeM_n2oPwxuQGP0pO6YnlLuOhvy6GxAY9QVpUuP5N6ChBASY2kkd_JcZ7lj9Tpsr4zPJahBrbhyENgt3A2IfGz5Y6SJzMXPtV4uhxy0vDuPsE2ARTHC8ia8QyM3nWubDg8TkB4yJRocU3X1NBO848bK4uF835J35o2fVaRNpBok16mK_laxqsUwPr7LRRX6Juoh-czUnybvqA90X5mP0q5gubcNtGsNgYxpDsrbcs1NtXXftPq80H2ZqvSaa6cZPFQ-rxQ0m0NAM2cO68aIq056ptFLI3J9LkRhjxGXb4XOrkahiDgGq-tFGHek7i7RhAW9SaMymjmMVw7p0-YXSclQ2bScpTqE_3AFRhYmYnAyGYu3Ae_DBSvqDkfR0j0el5Fb2ytTISwoFXvWC64xsl46t747_EBUuAZQWWxLKYAgBs9oQGTSInxaGSmHWkHXi5gVx6puNXSLSgTUNiSfR7nqBQLA4sCGcfXnPEints3InpgB_vXYiEq5c2cCpR0ETTSsd7ULPeZeUo5FBgFwjPeWkgKiBkPGHLk3BXNzkyvoIKRaE-tAceyOWa3CDoDhviggUOT8iJk7BNcOiJInxZJZ-PRHLIHz3tzuV-kJ_1PVO7NPlmfiL7Dvoaw77PdGyPgK9P-XND5lJpyHlqsFut3khyK31gMljXxjn9LwLawHLo_laaA1p17RAGmdwWAvR9orFV9fbWVTByyzt0MHtfQBk2ZydtJidlADpXfupzaa9V2geiUWFjMv2c4uQ0F_zeJfyGuQN1zCNk00E_U6hf3Z7r3BI6N81RXf-W_qTAOFtyOuF89DiNc", 
    app_key="m9q1908qk1ljaf6", 
    app_secret="pfjcdxadijcm0pg", 
    timeout = None,
)

paths = [
    # "tsp50_test_concorde.txt",
    # "tsp50_train_concorde.txt",
    # "tsp100_test_concorde.txt",
    # "tsp100_train_concorde.txt",
    # "tsp500_test_concorde.txt",
    # "tsp500_train_concorde.txt",
    # "tsp1000_test_concorde.txt",
    "tsp1000_train_lkh3.txt",
    "tsp10000_test_concorde.txt",
    "tsp10000_train_lkh3.txt"
    ]

for path in paths:
    dbx.files_download_to_file(f"/data/{path}", f"/TSPDataset/{path}")
