function process_channels(good_channels, path)
    for i = 1:length(good_channels)
        channel = good_channels(i);
        fichier_mat = fullfile(path, ['C' num2str(channel) '.mat']);
        fichier_spikes = fullfile(path, ['C' num2str(channel) '_spikes.mat']);
        fichier_times = fullfile(path, ['times_C' num2str(channel) '.mat']);

        % Exécuter Get_spikes
        disp(['Traitement du canal ' num2str(channel) ' : Exécution de Get_spikes...']);
        Get_spikes(fichier_mat);

        % Attendre que le fichier de spikes soit généré
        while ~isfile(fichier_spikes)
            disp(['En attente de la création de ' fichier_spikes '...']);
            pause(0.5);  % Attendre 0.5 secondes avant de vérifier à nouveau
        end

        % Exécuter Do_clustering
        disp(['Traitement du canal ' num2str(channel) ' : Exécution de Do_clustering...']);
        Do_clustering(fichier_spikes);

        % Attendre que le fichier times_C soit créé
        while ~isfile(fichier_times)
            disp(['En attente de la création de ' fichier_times '...']);
            pause(0.5);
        end

        disp(['Traitement du canal ' num2str(channel) ' terminé avec succès.']);
    end
end
