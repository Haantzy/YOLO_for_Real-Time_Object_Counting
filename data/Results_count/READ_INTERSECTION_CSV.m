Files = dir('*.csv');
for k = 1:length(Files)
    clf;
    [~,name,~] = fileparts(Files(k).name);
    
    data = readmatrix(Files(k).name);
    
    %Plot the traffic
    figure(1);
    for idx = 1:max(data(:,1))
        subplot(max(data(:,1)),1,idx)
    
        valid = data(:,1) == idx;
        data(valid,:);
        plot(data(valid,2),data(valid,3));
        
        grid ON
        title(sprintf('Number of cars crossing intersection: %d',idx))
        xlabel('Frame Number');
        ylabel('Num. Cars')
        ylim([-0.5 max(data(valid,3))+1])
    end
    
    saveas(gcf,sprintf('%s%s.%s',name,'_Results','jpg'))
   
end