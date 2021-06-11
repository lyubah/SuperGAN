#!/usr/bin/env ruby
# Usage: extract_labels.rb dataset_name

NUM_LABELS = {
  'adlnormal' => 5,
  'gyroscope' => 9,
  'accelerometer' => 9
}

# Grab the STS from the last epoch.
# This is consistently located four lines from the bottom
def get_sts filename
  f = File.open(filename, 'r')
  lines = f.each_line.to_a
  f.close
  lines[-5][16..-2].to_f
end

# Extract STS from all files and make a table
def make_table dataset_name
  range = 0 .. NUM_LABELS[dataset_name] - 1

  print ","
  range.each { |i| print "#{i}," }
  puts ""

  ["", "R", "C", "CR"].each do |kind|
    print "#{kind},"
    range.each do |i|
      filename = "stdout/#{dataset_name}_#{i}#{kind}.txt"
      sts = get_sts filename
      print "#{sts},"
    end
    puts ""
  end
end

make_table ARGV[0]