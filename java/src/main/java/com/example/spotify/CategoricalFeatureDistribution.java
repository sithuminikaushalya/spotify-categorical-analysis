package com.example.spotify;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

import java.io.IOException;
import java.io.StringReader;
import java.net.URISyntaxException;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;

public class CategoricalFeatureDistribution {

    // Mapper Class
    public static class DistributionMapper extends Mapper<LongWritable, Text, Text, IntWritable> {

        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();
        private boolean isHeader = true; // Flag to skip header row

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            // Skip the header row (assuming key.get() == 0 for the first line)
            if (key.get() == 0 && isHeader) {
                isHeader = false;
                return;
            }
            isHeader = false;

            String line = value.toString();

            try (StringReader reader = new StringReader(line)) {
                for (CSVRecord record : CSVFormat.DEFAULT.parse(reader)) {
                    // Check if the record has enough fields based on the highest index we need (21)
                    if (record.size() > 21) {
                        try {
                            // Extract 'key'
                            String keyValue = record.get(11).trim();
                            word.set("key:" + keyValue);
                            context.write(word, one);

                            // Extract 'mode'
                            String modeValue = record.get(13).trim();
                            // Convert mode to readable string (0: minor, 1: major), handling potential float conversion from original data
                            String modeString;
                            if (modeValue.equals("0") || modeValue.equals("0.0")) { // Handle "0" or "0.0"
                                modeString = "mode:minor";
                            } else if (modeValue.equals("1") || modeValue.equals("1.0")) { // Handle "1" or "1.0"
                                modeString = "mode:major";
                            } else {
                                modeString = "mode:unknown"; // Fallback for unexpected values
                            }
                            word.set(modeString);
                            context.write(word, one);

                            // Extract 'time_signature'
                            String timeSignatureValue = record.get(21).trim();
                            // It's often an integer, but the sample shows 4.0, so want to clean
                            if (timeSignatureValue.endsWith(".0")) {
                                timeSignatureValue = timeSignatureValue.substring(0, timeSignatureValue.length() - 2);
                            }
                            word.set("time_signature:" + timeSignatureValue);
                            context.write(word, one);

                        } catch (ArrayIndexOutOfBoundsException e) {
                            System.err.println("Skipping malformed record (column index out of bounds): " + line + " - Error: " + e.getMessage());
                        } catch (Exception e) {
                            System.err.println("Skipping bad record: " + line + " - General Parsing Error: " + e.getMessage());
                        }
                    } else {
                        System.err.println("Skipping record with insufficient columns after parsing: " + line + " (found " + record.size() + " columns)");
                    }
                }
            } catch (IOException e) {
                System.err.println("Error reading or parsing CSV line: " + line + " - " + e.getMessage());
            }
        }
    }

    // Reducer Class
    public static class DistributionReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private IntWritable result = new IntWritable();

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    // Driver Code
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();

        if (otherArgs.length != 2) {
            System.err.println("Usage: CategoricalFeatureDistribution <in> <out>");
            System.exit(2);
        }

        Job job = Job.getInstance(conf, "Categorical Feature Distribution");

        try {
            String jarPath = CategoricalFeatureDistribution.class.getProtectionDomain().getCodeSource().getLocation().toURI().getPath();
            job.setJar(jarPath);
        } catch (URISyntaxException e) {
            System.err.println("Error getting JAR path: " + e.getMessage());
            System.exit(1);
        }
        job.setJarByClass(CategoricalFeatureDistribution.class);

        job.setMapperClass(DistributionMapper.class);
        job.setCombinerClass(DistributionReducer.class);
        job.setReducerClass(DistributionReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
        FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}