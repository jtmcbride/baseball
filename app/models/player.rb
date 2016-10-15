class Player < ActiveRecord::Base
  self.table_name = 'master'
  has_many :battings, foreign_key: 'playerid'
  has_many :teams, through: :battings, source: :team
  has_many :teammates, -> { distinct }, through: :teams, source: :players
  # has_many :babe_ruth_numbers, through: :teams


  def name
    "#{self.namefirst} #{self.namelast}"
  end

  def method_missing(method)
    if method.to_s.starts_with?("total_career_")
      stat = method[13..-1]
      self.battings.sum(stat)
    end
  end

  def number
    n = BabeRuthNumber.find_by_player_id(@playerid)
    if n
      nil
    else
      n = nil
      self.teams.each do |team|
        nearest = BabeRuthNumber.find_by_team_id(team.id)
        n  = nearest if n == nil || nearest.distance < n.distance
      end
    end
    if n
      distance = n.distance
      answer = "#{self.name} played with "
      until n.distance == 0
        answer += "#{n.player.name} on the #{n.team.yearid} #{n.team.teamid} who played with "
        n.player.teams.each do |team|
          cand = BabeRuthNumber.find_by_team_id(team.id)
          n = cand if cand.distance < n.distance
          puts n.player.namelast
        end
      end
    else
      answer = nil
    end
    answer += "#{n.player.name} on #{n.team.yearid} #{n.team.teamid}. #{distance} steps away."
  end

end
