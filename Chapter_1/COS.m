function defuzzfun = COSdef(xmf, ymf)

    %xmf: πίνακας που περιέχει τιμές του πεδίου της mf
    %ymf: membership Value of x
    [pks, locs] = findpeaks(ymf, xmf); %Εύρεση των peaks και της τοποθεσίας τους (https://www.mathworks.com/help/signal/ref/findpeaks.html)
    synoliko_athroisma_perioxis = 0; %Σταθμησμένο άθροισμα εμβαδού
    athroisma_perioxis = 0; %Άθροισμα Εμβαδού
 
    %Υπολογισμός του COS με βάση τους τύπους για τις κεντρικές περιοχές
    for i = 1:length(pks)
        
        arxi = find(xmf == locs(i)); 
        thesi_mf = length(ymf(ymf == ymf(arxi)));% Βρίσκει το που είναι η mf 
        telos = arxi + thesi_mf - 1; % Υπολογίζει τον αριθμό που εμφανίζεται η Mf που έχουμε τώρα 
        anw_vasi = abs(xmf(telos) - xmf(arxi)); % Υπολογίζει το μήκος της μικρής βάσης του τραπεζοειδούς
        perioxi = 0.5*(1/4 + anw_vasi) * pks(i); % Τύπος για την εύρεση της επιφάνειας του τραπεζοειδούς
        kentro = 0.5*(xmf(telos) + xmf(arxi) ); %Τύπος για την εύρεση του κέντρου
        athroisma_perioxis = athroisma_perioxis + perioxi; % Πρόσθεση των προηγούμενων περιοχών με τις επόμενες
        synoliko_athroisma_perioxis = synoliko_athroisma_perioxis + perioxi * kentro; %Υπολογιζμός του αθροίσματος
    end
    
    %Λύση του προβλήματος με τα PV και NV
    
    %Λύση για το NV
     if (ymf(1) ~= 0) 
        
        %Εύρεση της θέσης της MF
        arxi = 1;
        thesi_mf = length(ymf(ymf == ymf(arxi)));
        telos = arxi + thesi_mf - 1; % Υπολογίζει τον αριθμό που εμφανίζεται η Mf που έχουμε τώρα 
        
        %Εύρεση της πάνω βάσης και της επιφάνειας του τραπεζοειδούς
        anw_vasi = abs(xmf(telos) - xmf(arxi)); % Υπολογισμός της πάνω βάσης του τραπεζοιδούς
        perioxi = 0.5*(1/4 + anw_vasi) * ymf(1); % Τύπος για την εύρεση της επιφάνειας του τραπεζοιδούς
        
        %Συναρτήσεις ευθειών για τον υπολογισμό των ολοκληρωμάτων
        y1 =@(w) (ymf(1) .*w); % Συνάρτηση υπολογισμού της ευθείας του τραπεζοειδούς
        y2 =@(w) ((ymf(1) / (xmf(telos) + 0.5)) .* (w + 0.5) .*w); % y = ax + b, με τιμές για τα α και β
        
        %Υπολογισμός ολοκληρωμάτων
        int_1 = integral(y1, - 1, xmf(telos)); %Υπολογισμός των απαραίτητων ολοκληρωμάτων
        int_2 = integral(y2, xmf(telos), - 0.5);
        int_total =int_1+int_2;
        
        %Τελικοί Υπολογισμοί για την εύρεση του σταθμισμένου μέσου όρου
        kentro =  int_total / perioxi;
        athroisma_perioxis = athroisma_perioxis + perioxi;
        synoliko_athroisma_perioxis = synoliko_athroisma_perioxis + kentro * perioxi;
        
     end

    %Λύση για το PV
    if (ymf(end) ~= 0) 
        
        %Εύρεση της θέσης της MF
        arxi = 101;
        thesi_mf = length(ymf(ymf == ymf(arxi)));
        telos = arxi - thesi_mf - 1; 
        
        %Εύρεση της πάνω βάσης και της επιφάνειας του τραπεζοειδούς        
        anw_vasi = abs( xmf(arxi)- xmf(telos));
        perioxi = 0.5*(1/4 + anw_vasi) * ymf(end);
        
        %Συναρτήσεις ευθειών για τον υπολογισμό των ολοκληρωμάτων
        y1 =@(w) ((ymf(end) / (xmf(telos) - 0.50)) .* (w - 0.50) .* w);
        y2 =@(w) (ymf(end) .*w);
        
        %Υπολογισμός ολοκληρωμάτων
        int_1 = integral(y1, 0.5, xmf(telos));
        int_2 = integral(y2, xmf(telos), 1);
        int_total =int_1+int_2;
        
        %Τελικοί Υπολογισμοί για την εύρεση του σταθμισμένου μέσου όρου
        kentro =  int_total / perioxi;
        athroisma_perioxis = athroisma_perioxis + perioxi;
        synoliko_athroisma_perioxis = synoliko_athroisma_perioxis + kentro * perioxi;
        
    end

    defuzzfun = synoliko_athroisma_perioxis / athroisma_perioxis;
    
end