function cost = DAB_Objective(X)
    alpha = X(1);
    phi1  = X(2);
    phi2  = X(3);

    assignin('base', 'alpha', alpha);
    assignin('base', 'phi1', phi1);
    assignin('base', 'phi2', phi2);

    try
        simOut = sim('DAB_Model', ...
                     'StopTime','1', ...
                     'SrcWorkspace','base', ...
                     'ReturnWorkspaceOutputs','on', ...
                     'SaveFormat','Dataset');

        logsout = simOut.logsout;

        Irms   = logsout.getElement(strtrim('Irms')).Values.Data(end);
        IPeak  = logsout.getElement(strtrim('IPeak')).Values.Data(end);
        P_loss = logsout.getElement(strtrim('P_loss')).Values.Data(end);

        cost = 0.6 * P_loss + 0.3 * Irms + 0.1 * IPeak;
    catch
        cost = 1e10;  % penalize failed runs
    end
end