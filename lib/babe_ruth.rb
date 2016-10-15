numbers = BabeRuthNumber.where(distance: 4)

numbers.each do |num|
  num.team.players.each do |player|
    player.teams.each do |team|
      x = BabeRuthNumber.new(player_id: player.playerid, team_id: team.id, distance: 5)
      if x.valid?
        x.save
      end
    end
  end
end
