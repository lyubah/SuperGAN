#!/usr/bin/env ruby

["", "R", "C", "CR"].each do |c|
  total_epochs = 0.0
  total_accuracy = 0.0

  datasets = { "adlnormal" => 4, "accelerometer" => 8, "gyroscope" => 8 }
  datasets.each do |name, n|
    (0..n).each do |i|
      File.open("stdout/#{name}_#{i}#{c}.txt", 'r') do |file|
        lines = file.each_line.to_a
        epoch = lines[-11][37..-32].to_i
        acc = lines[-7][40..-2].to_f

        total_epochs += epoch
        total_accuracy += acc
      end
    end
  end

  puts "Configuration: #{c.inspect}, avg accuracy: #{total_accuracy / 23.0}, avg epochs: #{total_epochs / 23.0}"
end