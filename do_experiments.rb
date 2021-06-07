#!/usr/bin/env ruby

require 'open3'
require 'fileutils'

MODEL_CONF_BASE = <<ENDTOML
[TRAINING_PARAMETERS]
latent_dimension = 10
epochs = 300
batch_size = 25
test_size = 100
real_synthetic_ratio = 5
synthetic_synthetic_ratio = 10
discriminator_learning_rate = 0.01
accuracy_threshold = 0.85
num_features = 9

[WEIGHTS]
discriminator_loss_weight = 1
classifier_loss_weight = 1
sfd_loss_weight = 1

[NAMES]
classifier_name = C

[MODELS]
directory = models
exists = False
ENDTOML

# Mapping of dataset names to the available labels
DATASET_LABELS = {
  'adlnormal' => [1, 2, 3, 4, 5],
  'gyroscope' => [0, 1, 2, 3, 4, 5, 6, 7, 8],
  'accelerometer' => [0, 1, 2, 3, 4, 5, 6, 7, 8]
}

class Experiment
  @@experiment_number = 0

  def initialize dataset_name, class_name, is_classifier, is_regularizer
    @dataset_name = dataset_name
    @class_name = class_name
    @is_classifier = is_classifier
    @is_regularizer = is_regularizer

    @lstm_file = "LSTM_#{@dataset_name}.h5"
    @dataset_file = case @dataset_name
                    when 'adlnormal' then 'CASAS_adlnormal_dataset.h5'
                    when 'accelerometer' then 'sports_data_accelerometer.h5'
                    when 'gyroscope' then 'sports_data_gyroscope.h5'
                    else ''
                    end
    @@experiment_number += 1                  
  end

  # Name extension for all files generated in this experiment
  def create_name_extension
    c = (@is_classifier ? 'C' : '')
    r = (@is_regularizer ? 'R' : '')
    "#{@dataset_name}_#{@class_name}#{c}#{r}"
  end

  # create a new model.conf to indicate where to save GAN models
  def write_model_conf
    extension = create_name_extension
    model_text = MODEL_CONF_BASE.clone
    model_text << "generator_filename = G_#{extension}.h5\n"
    model_text << "discriminator_filename = D_#{extension}.h5\n"
  
    File.open('model.conf', 'w') do |file|
      file.write model_text
    end       
  end

  # Create a .toml file to configure experiment
  def write_toml_file
    extension = create_name_extension
    toml_text = ""

    toml_text << "data_file_path = \"#{@dataset_file}\"\n"
    toml_text << "classifier_path = \"#{@lstm_file}\"\n"
    toml_text << "class_label = #{@class_name}\n"
    toml_text << "write_train_results = false\n"

    @toml_file = "#{extension}.toml"
    File.open(@toml_file, 'w') do |file|
      file.write toml_text
    end
  end

  # Setup relevant files, run SuperGAN, and then cleanup.
  def run_experiment
    write_model_conf
    write_toml_file

    # Generate command line parameters
    c = (@is_classifier ? '' : '-C ')
    r = (@is_regularizer ? '' : '-R ')

    extension = create_name_extension
    puts "RUNNING EXPERIMENT \##{@@experiment_number} : #{extension}"

    # Run command and send stdout to a file
    command = "python3 main.py #{@toml_file} #{c}#{r}--save"
    stdout, stderr, status = Open3.capture3(command)
    File.open("stdout/#{extension}.txt", 'w') do |file|
      file.write stdout
    end

    puts "RESULTING STATUS: #{status}"
    unless status.success? then
      File.open("stderr/#{extension}.txt", 'w') do |file|
        file.write stderr
      end
    end

    # Cleanup
    FileUtils.rm @toml_file
  end
end

DATASET_LABELS.each do |name, labels|
  labels.each do |label|
    Experiment.new(name, label, true, true).run_experiment
    Experiment.new(name, label, true, false).run_experiment
    Experiment.new(name, label, false, true).run_experiment
    Experiment.new(name, label, false, false).run_experiment
  end
end
