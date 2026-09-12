
clc;
clear all;
close all;

counter = 0;

%% read, plot, filter, and transform the EEGs
for sub = 1:9 % for no of subjects
    subject_no = strcat('A0', num2str(sub), 'E.mat');
    data = load(subject_no);

    save_path = ['D:\Paper_3_MI\datasets\BCICIV_2a_Exp\2D_CWT\sub_', num2str(sub), '\E\'];

    start = 4;
    % if sub == 4
    %     start = 4;
    % end
    for run = start:length(data.data) % for no of runs in a subject

        run_no = data.data{1,run}.X; % X has EEG signals of all 22 channels
        for trail = 1:47 % for no of trials in a run
            ind = [sub run trail]
            trial_start = data.data{1,run}.trial(trail);
            trial_stop = data.data{1,run}.trial(trail+1)-1;

            chan_22_mean = mean(run_no((trial_start:trial_stop), (1:22)))';

            miu_band = bandpass(chan_22_mean,[8 13]);
            beta_band = bandpass(chan_22_mean,[13 30]);
            gamma_band = bandpass(chan_22_mean,[33 200]);

            miu_band_amp = miu_band.*chan_22_mean;
            beta_band_amp = beta_band.*chan_22_mean;
            gamma_band_amp = gamma_band.*chan_22_mean;

            [wt,f] = cwt(miu_band_amp);
            L = length(miu_band_amp);
            t = 0:L-1;
            hp = pcolor(t,f,abs(wt));
            hp.EdgeColor = 'none';
            set(gca,'yscale','log');
            fname = sprintf('%d.png', counter);
            save_folder = fullfile(save_path, fname);
            saveas(gcf, save_folder)
            clear gca;

            [wt,f] = cwt(beta_band_amp);
            L = length(beta_band_amp);
            t = 0:L-1;
            hp = pcolor(t,f,abs(wt));
            hp.EdgeColor = 'none';
            set(gca,'yscale','log');
            save_folder = fullfile(save_path, fname+1);
            saveas(gcf, save_folder)
            clear gca;

            [wt,f] = cwt(gamma_band_amp);
            L = length(gamma_band_amp);
            t = 0:L-1;
            hp = pcolor(t,f,abs(wt));
            hp.EdgeColor = 'none';
            set(gca,'yscale','log');
            save_folder = fullfile(save_path, fname+1);
            saveas(gcf, save_folder)
            clear gca;
        end
    end
end


%% Loading CWTs, convert to gray and concatenate them
dist_path = 'D:\Paper_3_MI\datasets\BCICIV_2a_Exp\2D_CWT\sub_E\concat';
loading_path = 'D:\Paper_3_MI\datasets\BCICIV_2a_Exp\2D_CWT\sub_E\';
directory = dir(fullfile(loading_path, '/*.png'));
                count = numel(directory);
for i = 1:count
    source_img1 = imread(strcat(loading_path, '\', i));
    gray1 = rgb2gray(source_img1);
    
    source_img2 = imread(strcat(loading_path, '\', i+1));
    gray2 = rgb2gray(source_img2);

    source_img3 = imread(strcat(loading_path, '\', i+1));
    gray3 = rgb2gray(source_img3);

    concat = cat(3, gray1, gray2, gray3);

    fname1 = sprintf(i);
    dist_folder3 = fullfile(dist_path,fname1);
    imwrite(concat, dist_folder3)
end
