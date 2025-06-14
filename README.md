# Spotify Music Trends Analysis with Hadoop MapReduce

This project demonstrates how to deploy a Hadoop cluster on AWS using Terraform, process a large Spotify music dataset using a custom MapReduce job, and extract insights into categorical music features.

## Table of Contents

1.  [Project Overview](https://www.google.com/search?q=%231-project-overview)
2.  [Prerequisites](https://www.google.com/search?q=%232-prerequisites)
3.  [Deploying Hadoop Cluster on AWS (Terraform)](https://www.google.com/search?q=%233-deploying-hadoop-cluster-on-aws-terraform)
4.  [Setting Up Data on Master Node](https://www.google.com/search?q=%234-setting-up-data-on-master-node)
      * [Connecting to the Master Node](https://www.google.com/search?q=%23connecting-to-the-master-node)
      * [Setting up Kaggle API](https://www.google.com/search?q=%23setting-up-kaggle-api)
      * [Downloading and Uploading Dataset to HDFS](https://www.google.com/search?q=%23downloading-and-uploading-dataset-to-hdfs)
5.  [Running the MapReduce Job](https://www.google.com/search?q=%235-running-the-mapreduce-job)
      * [Clone and Build MapReduce Code](https://www.google.com/search?q=%23clone-and-build-mapreduce-code)
      * [Execute the Job](https://www.google.com/search?q=%23execute-the-job)
6.  [Viewing Results](https://www.google.com/search?q=%236-viewing-results)
8.  [References](https://www.google.com/search?q=%238-references)

-----

## 1\. Project Overview

This project focuses on analyzing a large dataset of Spotify music to understand the distribution of its fundamental musical characteristics: `key`, `mode`, and `time_signature`. It utilizes a Hadoop MapReduce job for distributed processing, deployed on an AWS cluster provisioned with Terraform.

## 2\. Prerequisites

Before you begin, ensure you have the following:

  * **AWS Account:** With sufficient permissions to create EC2 instances, VPCs, subnets, and security groups.
  * **AWS CLI Configured:** Ensure your AWS CLI is set up with credentials and a default region.
  * **Terraform Installed:** [Download & Install Terraform](https://learn.hashicorp.com/tutorials/terraform/install-cli)
  * **Java Development Kit (JDK) 8 or higher:** Installed on your local machine (for building the JAR) and will be installed on AWS instances via setup scripts.
  * **Apache Maven Installed:** [Download & Install Maven](https://maven.apache.org/install.html)
  * **Kaggle Account and API Key:** [Create a Kaggle account](https://www.kaggle.com/) and generate an API key (download `kaggle.json` from `Account -> API -> Create New API Token`).
  * **Git Installed:** For cloning repositories.

## 3\. Deploying Hadoop Cluster on AWS (Terraform)

This step will provision your Hadoop master and worker nodes, along with the necessary networking and security configurations, on AWS.

1.  **Clone the Terraform Scripts Repository:**

    ```bash
    git clone https://github.com/lakpahana/hadoop-terraform.git
    cd hadoop-terraform
    ```

2.  **Review Variables (Optional):**
    Inspect `variables.tf` to understand the configurable parameters (e.g., `aws_region`, `instance_type`, `worker_count`). You can create a `terraform.tfvars` file to override defaults if needed.

    ```hcl
    # Example terraform.tfvars (create this file if you need to override defaults)
    # aws_region = "us-east-1"
    # instance_type = "t3.medium"
    # worker_count = 2
    # key_name = "your-aws-key-pair-name" # Ensure this key pair exists in your AWS account
    ```

3.  **Initialize Terraform:**

    ```bash
    terraform init
    ```

4.  **Review the Plan:**

    ```bash
    terraform plan
    ```

    This will show you the resources Terraform will create. Review it carefully.

5.  **Apply the Terraform Configuration:**

    ```bash
    terraform apply
    ```

    Type `yes` when prompted to confirm the deployment. This process may take several minutes as EC2 instances are launched and user data scripts execute.


## 4\. Setting Up Data on Master Node

After the cluster is up, you need to connect to the master node, set up the Kaggle API, download the dataset, and upload it to HDFS.

### Connecting to the Master Node

You will connect to the master node using SSH. The default user for most Ubuntu AMIs is `ubuntu`, but your setup scripts might configure `hduser`. Please try `hduser` first as per our discussions.

```bash
ssh -i /path/to/your-aws-key-pair.pem hduser@<MASTER_NODE_PUBLIC_IP>
```

*Replace `/path/to/your-aws-key-pair.pem` with your actual SSH key file path and `<MASTER_NODE_PUBLIC_IP>` with the IP obtained from `terraform output`.*

### Setting up Kaggle API

Once you are SSHed into the master node:

1.  **Install Kaggle API:**

    ```bash
    pip install kaggle
    ```

2.  **Configure Kaggle API Key:**
    Create the `.kaggle` directory:

    ```bash
    mkdir -p ~/.kaggle
    ```

    Copy your `kaggle.json` file (downloaded from Kaggle account settings) from your local machine to `~/.kaggle/` on the master node. You can do this using `scp` from your *local machine* before SSHing:

    ```bash
    scp -i /path/to/your-aws-key-pair.pem /path/to/local/kaggle.json hduser@<MASTER_NODE_PUBLIC_IP>:~/.kaggle/
    ```

    Then, back on the master node, set correct permissions:

    ```bash
    chmod 600 ~/.kaggle/kaggle.json
    ```

### Downloading and Uploading Dataset to HDFS

1.  **Download the "Spotify 1.2M Songs" dataset:**

    ```bash
    kaggle datasets download -d rodolfofigueroa/spotify-12m-songs
    ```

    This will download `spotify-12m-songs.zip` to your current directory on the master node.

2.  **Unzip the dataset:**

    ```bash
    unzip spotify-12m-songs.zip
    ```

    This will extract `tracks_features.csv` (and possibly other files).

3.  **Create HDFS Input Directory:**

    ```bash
    hdfs dfs -mkdir -p /user/hduser/spotify_input
    ```

4.  **Upload `tracks_features.csv` to HDFS:**

    ```bash
    hdfs dfs -put tracks_features.csv /user/hduser/spotify_input/
    ```

5.  **Verify HDFS Upload:**

    ```bash
    hdfs dfs -ls /user/hduser/spotify_input/
    hdfs dfs -cat /user/hduser/spotify_input/tracks_features.csv | head -n 5
    ```

    Confirm you see `tracks_features.csv` and its content.

## 5\. Running the MapReduce Job

Now, you will clone the MapReduce job's source code, build it, and submit it to the Hadoop cluster.

### Clone and Build MapReduce Code

1.  **Clone the MapReduce Code Repository:**

    ```bash
    git clone https://github.com/sithuminikaushalya/spotify-categorical-analysis.git
    cd spotify-categorical-analysis/java # Navigate to the project root
    ```

2.  **Build the MapReduce Job JAR:**

    ```bash
    mvn clean package
    ```

    This will compile your Java code and create a JAR file (e.g., `spotify-categorical-analysis-1.0-SNAPSHOT-jar-with-dependencies.jar`) in the `target/` directory.

### Execute the Job

1.  **Remove Previous Output Directory (if any):**
    Hadoop jobs will fail if the output directory already exists.

    ```bash
    hdfs dfs -rm -r /user/hduser/spotify_output_categorical_dist
    ```

2.  **Run the MapReduce Job:**

    ```bash
    yarn jar target/spotify-categorical-analysis-1.0-SNAPSHOT-jar-with-dependencies.jar \
    /user/hduser/spotify_input \
    /user/hduser/spotify_output_categorical_dist
    ```

    Monitor the console output for job progress. This may take several minutes for 1.2 million rows.

## 6\. Viewing Results

Once the MapReduce job completes successfully:

1.  **List Output Files:**

    ```bash
    hdfs dfs -ls /user/hduser/spotify_output_categorical_dist/
    ```

    You should see `_SUCCESS` and `part-r-00000` (or multiple `part-r-#####` files).

2.  **View Results Content:**

    ```bash
    hdfs dfs -cat /user/hduser/spotify_output_categorical_dist/part-r-00000
    ```

    This will display the distribution counts for `key`, `mode`, and `time_signature`.

## 8\. References

  * **Terraform Scripts:** [https://github.com/lakpahana/hadoop-terraform](https://github.com/lakpahana/hadoop-terraform)
  * **MapReduce Code, Results, and Documentation:** [https://github.com/sithuminikaushalya/spotify-categorical-analysis](https://github.com/sithuminikaushalya/spotify-categorical-analysis)
  * **"Spotify 1.2M Songs" Dataset:** [https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs?select=tracks\_features.csv](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs?select=tracks_features.csv)

-----