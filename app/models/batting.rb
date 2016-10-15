class Batting < ActiveRecord::Base
  self.table_name = 'batting'
  belongs_to :player, foreign_key: 'playerid'
  belongs_to :team, foreign_key: 'fk_teamid', primary_key: 'pk_teamid' 
end
